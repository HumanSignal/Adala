import asyncio
import logging
from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Literal,
    TypedDict,
    Iterable,
    Generator,
)
from functools import cached_property
from enum import Enum
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
from adala.runtimes.base import CostEstimate
from adala.utils.exceptions import ConstrainedGenerationError
from adala.utils.internal_data import InternalDataFrame
from adala.utils.parse import (
    parse_template,
    partial_str_format,
    TemplateChunks,
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


# TODO: consolidate these data models and unify our preprocessing for LLM input into one step RawInputModel -> PreparedInputModel
class TextMessageChunk(TypedDict):
    type: Literal["text"]
    text: str


class ImageMessageChunk(TypedDict):
    type: Literal["image"]
    image_url: Dict[str, str]


MessageChunk = Union[TextMessageChunk, ImageMessageChunk]

Message = Union[str, List[MessageChunk]]


def get_messages(
    # user prompt can be a string or a list of multimodal message chunks
    user_prompt: Message,
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


def resolve_litellm_model_and_provider(model_name: str, provider: str):
    """
    When using litellm.get_model_info() some models are accessed with their provider prefix
    while others are not.

    This helper function contains logic which resolves this for supported providers
    """
    if "/" in model_name:  # TODO handle models like vertex_ai/meta/llama ...
        model_name = model_name.split("/")[1]
    provider = provider.lower()
    if provider == "vertexai":
        provider = "vertex_ai"

    return model_name, provider


class InstructorClientMixin(BaseModel):
    
    instructor_mode: str = "json_mode"

    def _from_litellm(self, **kwargs):
        return instructor.from_litellm(litellm.completion, **kwargs)

    @cached_property
    def client(self):
        return self._from_litellm(mode=self.instructor_mode)


class InstructorAsyncClientMixin(InstructorClientMixin):
    def _from_litellm(self, **kwargs):
        return instructor.from_litellm(litellm.acompletion, **kwargs)


def handle_llm_exception(
    e: Exception, messages: List[Dict[str, str]], model: str, retries
) -> tuple[Dict, Usage]:
    """Handle exceptions from LLM calls and return standardized error dict and usage stats.

    Args:
        e: The caught exception
        messages: The messages that were sent to the LLM
        model: The model name
        retries: The retry policy object

    Returns:
        Tuple of (error_dict, usage_stats)
    """

    if isinstance(e, IncompleteOutputException):
        usage = e.total_usage
    elif isinstance(e, InstructorRetryException):
        usage = e.total_usage
        # get root cause error from retries
        e = e.__cause__.last_attempt.exception()
    else:
        # Approximate usage for other errors
        # usage = e.total_usage
        # not available here, so have to approximate by hand, assuming the same error occurred each time
        n_attempts = retries.stop.max_attempt_number
        prompt_tokens = n_attempts * litellm.token_counter(
            model=model, messages=messages[:-1]
        )  # response is appended as the last message
        # TODO a pydantic validation error may be appended as the last message, don't know how to get the raw response in this case
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        )
        # Catch case where the model does not return a properly formatted out
        # AttributeError is an instructor bug: https://github.com/instructor-ai/instructor/pull/1103
        # > AttributeError: 'NoneType' object has no attribute '_raw_response'
        if type(e).__name__ in {"ValidationError", "AttributeError"}:
            logger.error(f"Converting error to ConstrainedGenerationError: {str(e)}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            e = ConstrainedGenerationError()

        # the only other instructor error that would be thrown is IncompleteOutputException due to max_tokens reached

    return _log_llm_exception(e), usage


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
            partial_str_format(input_template, **record, **extra_fields),
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
        except Exception as e:
            dct, usage = handle_llm_exception(e, messages, self.model, retries)

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
                # seed=self.seed,
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
            lambda row: partial_str_format(input_template, **row, **extra_fields),
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
            if isinstance(response, Exception):
                messages = []  # TODO how to get these?
                dct, usage = handle_llm_exception(
                    response, messages, self.model, retries
                )
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
        response_model: Optional[Type[BaseModel]] = None,
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
            instructions_first: If True, instructions will be sent before input.

        Returns:
            Dict[str, str]: The processed record.
        """
        # Create a single-row DataFrame from the input record
        input_df = InternalDataFrame([record])

        # Use the batch_to_batch method to process the single-row DataFrame
        output_df = await self.batch_to_batch(
            input_df,
            input_template=input_template,
            instructions_template=instructions_template,
            output_template=output_template,
            extra_fields=extra_fields,
            field_schema=field_schema,
            instructions_first=instructions_first,
            response_model=response_model,
        )

        # Extract the single row from the output DataFrame and convert it to a dictionary
        return output_df.iloc[0].to_dict()

    @staticmethod
    def _get_prompt_tokens(string: str, model: str, output_fields: List[str]) -> int:
        user_tokens = litellm.token_counter(model=model, text=string)
        # FIXME surprisingly difficult to get function call tokens, and doesn't add a ton of value, so hard-coding until something like litellm supports doing this for us.
        #       currently seems like we'd need to scrape the instructor logs to get the function call info, then use (at best) an openai-specific 3rd party lib to get a token estimate from that.
        system_tokens = 56 + (6 * len(output_fields))
        return user_tokens + system_tokens

    @staticmethod
    def _get_completion_tokens(
        model: str, output_fields: Optional[List[str]], provider: str
    ) -> int:
        model, provider = resolve_litellm_model_and_provider(model, provider)
        max_tokens = litellm.get_model_info(
            model=model, custom_llm_provider=provider
        ).get("max_tokens", None)
        if not max_tokens:
            raise ValueError
        # extremely rough heuristic, from testing on some anecdotal examples
        n_outputs = len(output_fields) if output_fields else 1
        return min(max_tokens, 4 * n_outputs)

    @classmethod
    def _estimate_cost(
        cls,
        user_prompt: str,
        model: str,
        output_fields: Optional[List[str]],
        provider: str,
    ):
        prompt_tokens = cls._get_prompt_tokens(user_prompt, model, output_fields)
        completion_tokens = cls._get_completion_tokens(model, output_fields, provider)
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_cost = prompt_cost + completion_cost

        return prompt_cost, completion_cost, total_cost

    def get_cost_estimate(
        self,
        prompt: str,
        substitutions: List[Dict],
        output_fields: Optional[List[str]],
        provider: str,
    ) -> CostEstimate:
        try:
            user_prompts = [
                partial_str_format(prompt, **substitution)
                for substitution in substitutions
            ]
            cumulative_prompt_cost = 0
            cumulative_completion_cost = 0
            cumulative_total_cost = 0
            for user_prompt in user_prompts:
                prompt_cost, completion_cost, total_cost = self._estimate_cost(
                    user_prompt=user_prompt,
                    model=self.model,
                    output_fields=output_fields,
                    provider=provider,
                )
                cumulative_prompt_cost += prompt_cost
                cumulative_completion_cost += completion_cost
                cumulative_total_cost += total_cost
            return CostEstimate(
                prompt_cost_usd=cumulative_prompt_cost,
                completion_cost_usd=cumulative_completion_cost,
                total_cost_usd=cumulative_total_cost,
            )

        except Exception as e:
            logger.error("Failed to estimate cost: %s", e)
            return CostEstimate(
                is_error=True,
                error_type=type(e).__name__,
                error_message=str(e),
            )


class MessageChunkType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"


def split_message_into_chunks(
    input_template: str, input_field_types: Dict[str, MessageChunkType], **input_fields
) -> List[MessageChunk]:
    """Split a template string with field types into a list of message chunks.

    Takes a template string with placeholders and splits it into chunks based on the field types,
    preserving the text between placeholders.

    Args:
        input_template (str): Template string with placeholders, e.g. '{a} is a {b} is an {a}'
        input_field_types (Dict[str, MessageChunkType]): Dict mapping field names to their types
        **input_fields: Field values to substitute into template

    Returns:
        List[Dict[str, str]]: List of message chunks with appropriate type and content.
            Text chunks have format: {'type': 'text', 'text': str}
            Image chunks have format: {'type': 'image_url', 'image_url': {'url': str}}

    Example:
        >>> split_message_into_chunks(
        ...     '{a} is a {b} is an {a}',
        ...     {'a': MessageChunkType.TEXT, 'b': MessageChunkType.IMAGE_URL},
        ...     a='the letter a',
        ...     b='http://example.com/b.jpg'
        ... )
        [
            {'type': 'text', 'text': 'the letter a is a '},
            {'type': 'image_url', 'image_url': {'url': 'http://example.com/b.jpg'}},
            {'type': 'text', 'text': ' is an the letter a'}
        ]
    """
    # Parse template to get field positions and surrounding text
    parsed = parse_template(input_template)

    def add_to_current_chunk(
        current_chunk: Optional[MessageChunk], chunk: MessageChunk
    ) -> MessageChunk:
        if current_chunk:
            current_chunk["text"] += chunk["text"]
            return current_chunk
        else:
            return chunk

    # Build chunks by iterating through parsed template parts
    def build_chunks(
        parsed: Iterable[TemplateChunks],
    ) -> Generator[MessageChunk, None, None]:
        current_chunk: Optional[MessageChunk] = None

        for part in parsed:
            if part["type"] == "text":
                current_chunk = add_to_current_chunk(
                    current_chunk, {"type": "text", "text": part["text"]}
                )
            elif part["type"] == "var":
                field_value = part["text"]
                try:
                    field_type = input_field_types[field_value]
                except KeyError:
                    raise ValueError(
                        f"Field {field_value} not found in input_field_types"
                    )
                if field_type == MessageChunkType.TEXT:
                    # try to substitute in variable and add to current chunk
                    substituted_text = partial_str_format(
                        f"{{{field_value}}}", **input_fields
                    )
                    if substituted_text != field_value:
                        current_chunk = add_to_current_chunk(
                            current_chunk, {"type": "text", "text": substituted_text}
                        )
                    else:
                        # be permissive for unfound variables
                        current_chunk = add_to_current_chunk(
                            current_chunk,
                            {"type": "text", "text": f"{{{field_value}}}"},
                        )
                elif field_type == MessageChunkType.IMAGE_URL:
                    substituted_text = partial_str_format(
                        f"{{{field_value}}}", **input_fields
                    )
                    if substituted_text != field_value:
                        # push current chunk, push image chunk, and start new chunk
                        if current_chunk:
                            yield current_chunk
                        current_chunk = None
                        yield {
                            "type": "image_url",
                            "image_url": {"url": input_fields[field_value]},
                        }
                    else:
                        # be permissive for unfound variables
                        current_chunk = add_to_current_chunk(
                            current_chunk,
                            {"type": "text", "text": f"{{{field_value}}}"},
                        )

        if current_chunk:
            yield current_chunk

    return list(build_chunks(parsed))


class AsyncLiteLLMVisionRuntime(AsyncLiteLLMChatRuntime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and vision
    models to perform the skill.
    """

    def init_runtime(self) -> "Runtime":
        super().init_runtime()
        # model_name = self.model
        # if not litellm.supports_vision(model_name):
        #     raise ValueError(f"Model {self.model} does not support vision")
        return self

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
        input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    ) -> InternalDataFrame:
        """Execute batch of requests with async calls to OpenAI API"""

        if not response_model:
            raise ValueError(
                "You must explicitly specify the `response_model` in runtime."
            )

        input_field_types = input_field_types or defaultdict(
            lambda: MessageChunkType.TEXT
        )

        extra_fields = extra_fields or {}
        user_prompts = batch.apply(
            # TODO: remove "extra_fields" to avoid name collisions
            lambda row: split_message_into_chunks(
                input_template, input_field_types, **row, **extra_fields
            ),
            axis=1,
        ).tolist()

        # rest of this function is the same as AsyncLiteLLMChatRuntime.batch_to_batch

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
            if isinstance(response, Exception):
                messages = []  # TODO how to get these?
                dct, usage = handle_llm_exception(
                    response, messages, self.model, retries
                )
            else:
                resp, completion = response
                usage = completion.usage
                dct = to_jsonable_python(resp)

            # Add usage data to the response (e.g. token counts, cost)
            dct.update(_get_usage_dict(usage, model=self.model))

            df_data.append(dct)

        output_df = InternalDataFrame(df_data)
        return output_df.set_index(batch.index)

    # TODO: cost estimate


def get_model_info(
    provider: str, model_name: str, auth_info: Optional[dict] = None
) -> dict:
    if auth_info is None:
        auth_info = {}
    try:
        # for azure models, need to get the canonical name for the model
        if provider == "azure":
            dummy_completion = litellm.completion(
                model=f"azure/{model_name}",
                messages=[{"role": "user", "content": ""}],
                max_tokens=1,
                **auth_info,
            )
            model_name = dummy_completion.model
        model_name, provider = resolve_litellm_model_and_provider(model_name, provider)
        return litellm.get_model_info(model=model_name, custom_llm_provider=provider)
    except Exception as err:
        logger.error("Hit error when trying to get model metadata: %s", err)
        return {}
