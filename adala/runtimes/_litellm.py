import asyncio
import logging
from typing import Any, Dict, List, Optional, Type
from functools import cached_property
import litellm
from litellm.exceptions import (
    AuthenticationError,
    ContentPolicyViolationError,
    BadRequestError,
    NotFoundError,
)
from litellm.types.utils import Usage
import instructor
from instructor.exceptions import InstructorRetryException, IncompleteOutputException
import traceback
from adala.utils.exceptions import ConstrainedGenerationError
from adala.utils.internal_data import InternalDataFrame
from adala.utils.parse import (
    parse_template,
    partial_str_format,
)
from pydantic import ConfigDict, field_validator, BaseModel
from pydantic_core import to_jsonable_python
from rich import print
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from pydantic_core._pydantic_core import ValidationError

from .base import AsyncRuntime, Runtime

logger = logging.getLogger(__name__)


# basically only retrying on timeout, incomplete output, or rate limit
# https://docs.litellm.ai/docs/exception_mapping#custom-mapping-list
# NOTE: token usage is only correctly calculated if we only use instructor retries, not litellm retries
# https://github.com/jxnl/instructor/pull/763
RETRY_POLICY = dict(
    retry=retry_if_not_exception_type(
        (
            ValidationError,
            ContentPolicyViolationError,
            AuthenticationError,
            BadRequestError,
        )
    ),
    # should stop earlier on ValidationError and later on other errors, but couldn't figure out how to do that cleanly
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=60),
)


def get_messages(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    instruction_first: bool = True,
):
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        if instruction_first:
            messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages[0]["content"] += system_prompt
    return messages


def _format_error_dict(e: Exception) -> dict:
    error_message = type(e).__name__
    error_details = str(e)
    # TODO change this format?
    error_dct = {
        "_adala_error": True,
        "_adala_message": error_message,
        "_adala_details": error_details,
    }
    return error_dct


def _log_llm_exception(e) -> dict:
    dct = _format_error_dict(e)
    base_error = f"Inference error {dct['_adala_message']}"
    tb = "".join(
        traceback.format_exception(e)
    )  # format_exception return list of strings ending in new lines
    logger.error(f"{base_error}\nTraceback:\n{tb}")
    return dct


def _get_usage_dict(usage: Usage, model: str) -> Dict:
    data = dict()
    data["_prompt_tokens"] = usage.prompt_tokens

    # will not exist if there is no completion
    # sometimes the response will have a CompletionUsage instead of a Usage, which doesn't have a .get() method
    # data["_completion_tokens"] = usage.get("completion_tokens", 0)
    try:
        data["_completion_tokens"] = usage.completion_tokens
    except AttributeError:
        data["_completion_tokens"] = 0

    # can't use litellm.completion_cost bc it only takes the most recent completion, and .usage is summed over retries
    # TODO make sure this is calculated correctly after we turn on caching
    # litellm will register the cost of an azure model on first successful completion. If there hasn't been a successful completion, the model will not be registered
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model, data["_prompt_tokens"], data["_completion_tokens"]
        )
        data["_prompt_cost_usd"] = prompt_cost
        data["_completion_cost_usd"] = completion_cost
        data["_total_cost_usd"] = prompt_cost + completion_cost
    except NotFoundError:
        logger.error(f"Failed to get cost for model {model}")
        data["_prompt_cost_usd"] = None
        data["_completion_cost_usd"] = None
        data["_total_cost_usd"] = None
    return data


class InstructorClientMixin:

    def _from_litellm(self, **kwargs):
        return instructor.from_litellm(litellm.completion, **kwargs)

    @cached_property
    def client(self):
        kwargs = {}
        if self.is_custom_openai_endpoint:
            kwargs["mode"] = instructor.Mode.JSON
        return self._from_litellm(**kwargs)

    @property
    def is_custom_openai_endpoint(self) -> bool:
        return self.model.startswith("openai/") and self.model_extra.get("base_url")


class InstructorAsyncClientMixin(InstructorClientMixin):

    def _from_litellm(self, **kwargs):
        return instructor.from_litellm(litellm.acompletion, **kwargs)


class LiteLLMChatRuntime(InstructorClientMixin, Runtime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and chat
    completion models to perform the skill.

    The default model provider is [OpenAI](https://openai.com/), using the OPENAI_API_KEY environment variable. Other providers [can be chosen](https://litellm.vercel.app/docs/set_keys) through environment variables or passed parameters.

    Attributes:
        model: model name. Refer to litellm supported models for how to pass
               this: https://litellm.vercel.app/docs/providers
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.
        seed: Integer seed to reduce nondeterminism in generation.

    Extra parameters passed to this class will be used for inference. See `litellm.types.completion.CompletionRequest` for a full list. Some common ones are:
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url (Optional[str]): Base URL, optional. If provided, will be used to talk to an OpenAI-compatible API provider besides OpenAI.
        api_version (Optional[str]): API version, optional except for Azure.
        timeout: Timeout in seconds.
    """

    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.0
    seed: Optional[int] = 47

    model_config = ConfigDict(extra="allow")

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            messages = [{"role": "user", "content": "Hey, how's it going?"}]
            litellm.completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                # extra inference params passed to this runtime
                **self.model_extra,
            )
        except AuthenticationError:
            raise ValueError(
                f'Requested model "{self.model}" is not available with your api_key and settings.'
            )
        except Exception as e:
            raise ValueError(
                f'Failed to check availability of requested model "{self.model}": {e}'
            )

        return self

    def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        # TODO: sunset this method in favor of record_to_record
        if self.verbose:
            print(f"**Prompt content**:\n{messages}")
        completion = litellm.completion(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            # extra inference params passed to this runtime
            **self.model_extra,
        )
        completion_text = completion.choices[0].message.content
        if self.verbose:
            print(f"**Response**:\n{completion_text}")
        return completion_text

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        output_template: Optional[
            str
        ] = None,  # TODO: deprecated in favor of response_model, can be removed
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = False,
    ) -> Dict[str, str]:
        """
        Execute OpenAI request given record and templates for input, instructions and output.

        Args:
            record: Record to be used for input, instructions and output templates.
            input_template: Template for input message.
            instructions_template: Template for instructions message.
            output_template: Template for output message (deprecated, not used).
            extra_fields: Extra fields to be used in templates.
            field_schema: Field schema to be used for parsing templates.
            instructions_first: If True, instructions will be sent before input.
            response_model: Pydantic model for response.

        Returns:
            Dict[str, str]: Output record.
        """

        extra_fields = extra_fields or {}

        if not response_model:
            raise ValueError(
                "You must explicitly specify the `response_model` in runtime."
            )

        messages = get_messages(
            input_template.format(**record, **extra_fields),
            instructions_template,
            instructions_first,
        )

        retries = Retrying(**RETRY_POLICY)

        try:
            # returns a pydantic model named Output
            response, completion = self.client.chat.completions.create_with_completion(
                messages=messages,
                response_model=response_model,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                max_retries=retries,
                # extra inference params passed to this runtime
                **self.model_extra,
            )
            usage = completion.usage
            dct = to_jsonable_python(response)
        except IncompleteOutputException as e:
            usage = e.total_usage
            dct = _log_llm_exception(e)
        except InstructorRetryException as e:
            usage = e.total_usage
            # get root cause error from retries
            n_attempts = e.n_attempts
            e = e.__cause__.last_attempt.exception()
            dct = _log_llm_exception(e)
        except Exception as e:
            # usage = e.total_usage
            # not available here, so have to approximate by hand, assuming the same error occurred each time
            n_attempts = retries.stop.max_attempt_number
            prompt_tokens = n_attempts * litellm.token_counter(
                model=self.model, messages=messages[:-1]
            )  # response is appended as the last message
            # TODO a pydantic validation error may be appended as the last message, don't know how to get the raw response in this case
            completion_tokens = 0
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(prompt_tokens + completion_tokens),
            )

            # Catch case where the model does not return a properly formatted output
            if type(e).__name__ == "ValidationError" and "Invalid JSON" in str(e):
                e = ConstrainedGenerationError()
            # there are no other known errors to catch
            dct = _log_llm_exception(e)

        # Add usage data to the response (e.g. token counts, cost)
        dct.update(_get_usage_dict(usage, model=self.model))

        return dct


class AsyncLiteLLMChatRuntime(InstructorAsyncClientMixin, AsyncRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion
    models to perform the skill. It uses async calls to OpenAI API.

    The default model provider is [OpenAI](https://openai.com/), using the OPENAI_API_KEY environment variable. Other providers [can be chosen](https://litellm.vercel.app/docs/set_keys) through environment variables or passed parameters.

    Attributes:
        model: model name. Refer to litellm supported models for how to pass
               this: https://litellm.vercel.app/docs/providers
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.
        seed: Integer seed to reduce nondeterminism in generation.

    Extra parameters passed to this class will be used for inference. See `litellm.types.completion.CompletionRequest` for a full list. Some common ones are:
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url (Optional[str]): Base URL, optional. If provided, will be used to talk to an OpenAI-compatible API provider besides OpenAI.
        api_version (Optional[str]): API version, optional except for Azure.
        timeout: Timeout in seconds.
    """

    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.0
    seed: Optional[int] = 47

    model_config = ConfigDict(extra="allow")

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            messages = [{"role": "user", "content": "Hey, how's it going?"}]
            litellm.completion(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                # extra inference params passed to this runtime
                **self.model_extra,
            )
        except AuthenticationError:
            raise ValueError(
                f'Requested model "{self.model}" is not available with your api_key and settings.'
            )
        except Exception as e:
            raise ValueError(
                f'Failed to check availability of requested model "{self.model}": {e}'
            )

        return self

    @field_validator("concurrency", mode="before")
    def check_concurrency(cls, value) -> int:
        value = value or -1
        if value < 1:
            raise NotImplementedError(
                "You must explicitly specify the number of concurrent clients for AsyncOpenAIChatRuntime. "
                "Set `AsyncOpenAIChatRuntime(concurrency=10, ...)` or any other positive integer. "
            )
        return value

    async def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        output_template: Optional[
            str
        ] = None,  # TODO: deprecated in favor of response_model, can be removed
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        """Execute batch of requests with async calls to OpenAI API"""

        if not response_model:
            raise ValueError(
                "You must explicitly specify the `response_model` in runtime."
            )

        extra_fields = extra_fields or {}
        user_prompts = batch.apply(
            # TODO: remove "extra_fields" to avoid name collisions
            lambda row: input_template.format(**row, **extra_fields),
            axis=1,
        ).tolist()

        retries = AsyncRetrying(**RETRY_POLICY)

        tasks = [
            asyncio.ensure_future(
                self.client.chat.completions.create_with_completion(
                    messages=get_messages(
                        user_prompt,
                        instructions_template,
                        instructions_first,
                    ),
                    response_model=response_model,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=self.seed,
                    max_retries=retries,
                    # extra inference params passed to this runtime
                    **self.model_extra,
                )
            )
            for user_prompt in user_prompts
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # convert list of LLMResponse objects to the dataframe records
        df_data = []
        for response in responses:
            if isinstance(response, IncompleteOutputException):
                e = response
                usage = e.total_usage
                dct = _log_llm_exception(e)
            elif isinstance(response, InstructorRetryException):
                e = response
                usage = e.total_usage
                # get root cause error from retries
                n_attempts = e.n_attempts
                e = e.__cause__.last_attempt.exception()
                dct = _log_llm_exception(e)
            elif isinstance(response, Exception):
                e = response
                # usage = e.total_usage
                # not available here, so have to approximate by hand, assuming the same error occurred each time
                n_attempts = retries.stop.max_attempt_number
                messages = []  # TODO how to get these?
                prompt_tokens = n_attempts * litellm.token_counter(
                    model=self.model, messages=messages[:-1]
                )  # response is appended as the last message
                # TODO a pydantic validation error may be appended as the last message, don't know how to get the raw response in this case
                completion_tokens = 0
                usage = Usage(
                    prompt_tokens,
                    completion_tokens,
                    total_tokens=(prompt_tokens + completion_tokens),
                )

                # Catch case where the model does not return a properly formatted output
                if type(e).__name__ == "ValidationError" and "Invalid JSON" in str(e):
                    e = ConstrainedGenerationError()
                # the only other instructor error that would be thrown is IncompleteOutputException due to max_tokens reached
                dct = _log_llm_exception(e)
            else:
                resp, completion = response
                usage = completion.usage
                dct = to_jsonable_python(resp)

            # Add usage data to the response (e.g. token counts, cost)
            dct.update(_get_usage_dict(usage, model=self.model))

            df_data.append(dct)

        output_df = InternalDataFrame(df_data)
        return output_df.set_index(batch.index)

    async def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> Dict[str, str]:
        raise NotImplementedError("record_to_record is not implemented")


class LiteLLMVisionRuntime(LiteLLMChatRuntime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and vision
    models to perform the skill.
    """

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = False,
    ) -> Dict[str, str]:
        """
        Execute LiteLLM request given record and templates for input,
        instructions and output.

        Args:
            record: Record to be used for input, instructions and output templates.
            input_template: Template for input message.
            instructions_template: Template for instructions message.
            output_template: Template for output message.
            extra_fields: Extra fields to be used in templates.
            field_schema: Field jsonschema to be used for parsing templates.
                          Field schema must contain "format": "uri" for image fields.
                          For example:
                            ```json
                            {
                                "image": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                            ```
            instructions_first: If True, instructions will be sent before input.
        """

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        output_fields = parse_template(
            partial_str_format(output_template, **extra_fields),
            include_texts=False,
        )

        if len(output_fields) > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support multiple output fields. "
                f"Found: {output_fields}"
            )
        output_field = output_fields[0]
        output_field_name = output_field["text"]

        input_fields = parse_template(input_template)

        # split input template into text and image parts
        input_text = ""
        content = [
            {
                "type": "text",
                "text": instructions_template,
            }
        ]
        for field in input_fields:
            if field["type"] == "text":
                input_text += field["text"]
            elif field["type"] == "var":
                if field["text"] not in field_schema:
                    input_text += record[field["text"]]
                elif field_schema[field["text"]]["type"] == "string":
                    if field_schema[field["text"]].get("format") == "uri":
                        if input_text:
                            content.append({"type": "text", "text": input_text})
                            input_text = ""
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": record[field["text"]],
                            }
                        )
                    else:
                        input_text += record[field["text"]]
                else:
                    raise ValueError(
                        f'Unsupported field type: {field_schema[field["text"]]["type"]}'
                    )
        if input_text:
            content.append({"type": "text", "text": input_text})

        if self.verbose:
            print(f"**Prompt content**:\n{content}")

        completion = litellm.completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            # extra inference params passed to this runtime
            **self.model_extra,
        )

        completion_text = completion.choices[0].message.content
        return {output_field_name: completion_text}
