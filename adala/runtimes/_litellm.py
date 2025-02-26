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
    APIConnectionError,
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
    MessageChunkType,
)
from adala.utils.llm_utils import (
    run_instructor_with_payload,
    run_instructor_with_payloads,
    arun_instructor_with_payload,
    arun_instructor_with_payloads,
    run_instructor_with_messages,
    arun_instructor_with_messages,
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


def normalize_litellm_model_and_provider(model_name: str, provider: str):
    """
    When using litellm.get_model_info() some models are accessed with their provider prefix
    while others are not.

    This helper function contains logic which normalizes this for supported providers
    """
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]
    provider = provider.lower()
    if provider == "vertexai":
        provider = "vertex_ai"

    return model_name, provider


class InstructorClientMixin(BaseModel):

    # Note: most models work better with json mode; this is set only for backwards compatibility
    # instructor_mode: str = "json_mode"
    instructor_mode: str = "tool_call"

    # Note: doesn't seem like this separate function should be necessary, but errors when combined with @cached_property
    def _from_litellm(self, **kwargs):
        return instructor.from_litellm(litellm.completion, **kwargs)

    @cached_property
    def client(self):
        return self._from_litellm(mode=instructor.Mode(self.instructor_mode))


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
    """

    model: str = "gpt-4o-mini"
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
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

        retries = Retrying(**RETRY_POLICY)
        
        return run_instructor_with_payload(
            client=self.client,
            payload=record,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=retries,
            extra_fields=extra_fields,
            instructions_first=instructions_first,
            instructions_template=instructions_template,
            **self.model_extra,
        )


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
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
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
        
        # convert batch to list of payloads
        payloads = batch.to_dict(orient="records")
        retries = AsyncRetrying(**RETRY_POLICY)
        extra_fields = extra_fields or {}
        
        df_data = await arun_instructor_with_payloads(
            client=self.client,
            payloads=payloads,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=retries,
            extra_fields=extra_fields,
            instructions_first=instructions_first,
            instructions_template=instructions_template,
            **self.model_extra,
        )

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
        model, provider = normalize_litellm_model_and_provider(model, provider)
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


class AsyncLiteLLMVisionRuntime(AsyncLiteLLMChatRuntime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and vision
    models to perform the skill.
    """

    def init_runtime(self) -> "Runtime":
        super().init_runtime()
        # Only running this supports_vision check for non-vertex models, since its based on a static JSON file in
        # litellm which was not up to date. Will be soon in next release - should update this
        if not self.model.startswith("vertex_ai"):
            model_name = self.model
            if not litellm.supports_vision(model_name):
                raise ValueError(f"Model {self.model} does not support vision")
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

        extra_fields = extra_fields or {}
        records = batch.to_dict(orient="records")
        retries = AsyncRetrying(**RETRY_POLICY)
        
        df_data = await arun_instructor_with_payloads(
            client=self.client,
            payloads=records,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=retries,
            split_into_chunks=True,
            input_field_types=input_field_types,
            instructions_first=instructions_first,
            instructions_template=instructions_template,
            extra_fields=extra_fields,
            **self.model_extra,
        )

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
        model_name, provider = normalize_litellm_model_and_provider(
            model_name, provider
        )
        return litellm.get_model_info(model=model_name, custom_llm_provider=provider)
    except Exception as err:
        logger.error("Hit error when trying to get model metadata: %s", err)
        return {}
