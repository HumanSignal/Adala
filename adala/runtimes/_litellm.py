import asyncio
import logging
import traceback
from openai import OpenAI, AsyncOpenAI
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
from adala.utils.model_info_utils import (
    match_model_provider_string,
    NoModelsFoundError,
    _estimate_cost,
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


# safeguard for custom models where we don't know what params are supported
litellm.drop_params = True


# basically only retrying on timeout, incomplete output, or rate limit
# https://docs.litellm.ai/docs/exception_mapping#custom-mapping-list
# NOTE: token usage is only correctly calculated if we only use instructor retries, not litellm retries
# https://github.com/jxnl/instructor/pull/763
# RETRY_POLICY = dict(
#     retry=retry_if_not_exception_type(
#         (
#             ValidationError,
#             ContentPolicyViolationError,
#             AuthenticationError,
#             BadRequestError,
#         )
#     ),
#     # should stop earlier on ValidationError and later on other errors, but couldn't figure out how to do that cleanly
#     stop=stop_after_attempt(3),
#     wait=wait_random_exponential(multiplier=1, max=60),
# )

# For now, disabling all instructor retries as of DIA-1910 to speed up inference runs greatly
RETRY_POLICY = dict(stop=stop_after_attempt(1))
retries = Retrying(**RETRY_POLICY)
async_retries = AsyncRetrying(**RETRY_POLICY)


class InstructorClientMixin(BaseModel):

    # Note: most models work better with json mode; this is set only for backwards compatibility
    # instructor_mode: str = "json_mode"
    instructor_mode: str = "tool_call"
    provider: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @property
    def _litellm_client(self):
        return litellm.completion

    @property
    def _openai_client(self):
        return OpenAI

    def _check_client(self):
        # don't use response model and error handling from run_instructor_with_messages here
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            response_model=None,
            max_retries=retries,
            # extra inference params passed to this runtime
            **self.model_extra,
        )
        return response

    # Yes, this is recomputed every time - there's no clean way to cache it unless we drop pickle serialization, in which case adding it to ConfigDict(ignore) would work.
    # There's no appreciable startup cost in the instructor client init function anyway.
    @property
    def client(self):
        if self.provider == "Custom":
            logger.info(f"Custom provider: using OpenAI client.")
            return instructor.from_openai(
                self._openai_client(api_key=self.api_key, base_url=self.base_url),
                mode=instructor.Mode(self.instructor_mode),
            )
        return instructor.from_litellm(
            self._litellm_client,
            api_key=self.api_key,
            base_url=self.base_url,
            mode=instructor.Mode(self.instructor_mode),
        )

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            self._check_client()
        except AuthenticationError:
            logger.exception(
                f'Requested model "{self.model}" is not available with your api_key and settings.\nTraceback:\n{traceback.format_exc()}'
            )
            raise ValueError(
                f'Requested model "{self.model}" is not available with your api_key and settings.'
            )
        except Exception as e:
            logger.exception(
                f'Failed to check availability of requested model "{self.model}": {e}\nTraceback:\n{traceback.format_exc()}'
            )
            raise ValueError(
                f'Failed to check availability of requested model "{self.model}": {e}'
            )

        return self

    def get_canonical_model_provider_string(self, model: str) -> str:
        """provider_name/model_name"""
        # this is really a litellm function, not an instructor function. Putting it here to avoid duplicating it between sync/async runtimes.
        try:
            return match_model_provider_string(model)
        except NoModelsFoundError:
            logger.info(
                f"Model {model} not found in litellm model map for provider {self.provider}. This is likely a single-model deployment."
            )
        except Exception as e:
            logger.exception(
                f"(1/2) Failed to get canonical model provider string for {model}"
            )
        try:
            resp = self._check_client()
            return match_model_provider_string(resp.model)
        except NoModelsFoundError:
            logger.warning(
                f"Model {model} not found in litellm model map for provider {self.provider}. This is likely a custom model."
            )
            return model
        except Exception as e:
            logger.exception(
                f"(2/2) Failed to get canonical model provider string for {model}"
            )
            return model


class InstructorAsyncClientMixin(InstructorClientMixin):

    @property
    def _litellm_client(self):
        return litellm.acompletion

    @property
    def _openai_client(self):
        return AsyncOpenAI

    def _check_client(self):
        """Make this synchronous"""
        client = InstructorClientMixin(**self.model_dump()).client
        # don't use response model and error handling from run_instructor_with_messages here
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            response_model=None,
            max_retries=retries,
            # extra inference params passed to this runtime
            **self.model_extra,
        )
        return response


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

    @property
    def canonical_model_provider_string(self):
        """provider_name/model_name"""
        return self.get_canonical_model_provider_string(self.model)

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

    @field_validator("concurrency", mode="before")
    def check_concurrency(cls, value) -> int:
        value = value or -1
        if value < 1:
            raise NotImplementedError(
                "You must explicitly specify the number of concurrent clients for AsyncOpenAIChatRuntime. "
                "Set `AsyncOpenAIChatRuntime(concurrency=10, ...)` or any other positive integer. "
            )
        return value

    @property
    def canonical_model_provider_string(self):
        """provider_name/model_name"""
        return self.get_canonical_model_provider_string(self.model)

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
            retries=async_retries,
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
            model = self.canonical_model_provider_string
            for user_prompt in user_prompts:
                prompt_cost, completion_cost, total_cost = _estimate_cost(
                    user_prompt=user_prompt,
                    model=model,
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
        input_field_types = input_field_types or {}
        records = batch.to_dict(orient="records")
        # in multi-image cases, the number of tokens can be too large for the context window
        # so we need to split the payloads into chunks
        # we use this heuristic for MIG projects as they more likely to have multi-image inputs
        # for other data types, we skip checking the context window as it will be slower
        ensure_messages_fit_in_context_window = any(
            input_field_types.get(field) == MessageChunkType.IMAGE_URLS
            for field in input_field_types
        )

        df_data = await arun_instructor_with_payloads(
            client=self.client,
            payloads=records,
            user_prompt_template=input_template,
            response_model=response_model,
            model=self.canonical_model_provider_string,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            retries=async_retries,
            split_into_chunks=True,
            input_field_types=input_field_types,
            instructions_first=instructions_first,
            instructions_template=instructions_template,
            extra_fields=extra_fields,
            ensure_messages_fit_in_context_window=ensure_messages_fit_in_context_window,
            **self.model_extra,
        )

        output_df = InternalDataFrame(df_data)
        return output_df.set_index(batch.index)
