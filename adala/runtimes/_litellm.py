import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic_settings import BaseSettings
import litellm
from litellm.exceptions import AuthenticationError
import instructor
import traceback
from adala.utils.internal_data import InternalDataFrame
from adala.utils.logs import print_error
from adala.utils.parse import (
    parse_template,
    partial_str_format,
    parse_template_to_pydantic_class,
)
from pydantic import ConfigDict, field_validator
from rich import print

from .base import AsyncRuntime, Runtime

instructor_client = instructor.from_litellm(litellm.completion)
async_instructor_client = instructor.from_litellm(litellm.acompletion)

logger = logging.getLogger(__name__)


class LiteLLMInferenceSettings(BaseSettings):
    """
    Common inference settings for LiteLLM.

    See `litellm.types.completion.CompletionRequest` for other parameters not set here.

    Attributes:
        model: model name. Refer to litellm supported models for how to pass
               this: https://litellm.vercel.app/docs/providers
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url (Optional[str]): Base URL, optional. If provided, will be used to talk to an OpenAI-compatible API provider besides OpenAI.
        api_version (Optional[str]): API version, optional except for Azure.
        instruction_first: Whether to put instructions first.
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.
        timeout: Timeout in seconds.
        seed: Integer seed to reduce nondeterminism in generation.
    """

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.0
    timeout: Optional[Union[float, int]] = None
    seed: Optional[int] = 47


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


class LiteLLMChatRuntime(LiteLLMInferenceSettings, Runtime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and chat
    completion models to perform the skill.

    Attributes:
        inference_settings (LiteLLMInferenceSettings): Common inference settings for LiteLLM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    def as_inference_settings(self, **override_kwargs) -> LiteLLMInferenceSettings:
        inference_settings = LiteLLMInferenceSettings(
            **self.dict(include=LiteLLMInferenceSettings.model_fields.keys())
        )
        inference_settings.update(**override_kwargs)
        return inference_settings

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            messages = [{"role": "user", "content": "Hey, how's it going?"}]
            litellm.completion(
                messages=messages, **self.as_inference_settings(max_tokens=10).dict()
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
            **self.as_inference_settings().dict(),
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
        output_template: str,
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
            output_template: Template for output message.
            extra_fields: Extra fields to be used in templates.
            field_schema: Field schema to be used for parsing templates.
            instructions_first: If True, instructions will be sent before input.

        Returns:
            Dict[str, str]: Output record.
        """

        extra_fields = extra_fields or {}

        response_model = parse_template_to_pydantic_class(
            output_template, provided_field_schema=field_schema
        )
        messages = get_messages(
            input_template.format(**record, **extra_fields),
            self.system_prompt,
            self.instruction_first,
        )

        try:
            response = instructor_client.chat.completions.create(
                messages=messages,
                response_model=response_model,
                **self.as_inference_settings().dict(),
            )
        except Exception as e:
            error_message = type(e).__name__
            # error_details = str(e)
            error_details = traceback.format_exc()
            if self.verbose:
                print_error(error_message, error_details)
            # TODO change this format
            error_dct = {
                "_adala_error": True,
                "_adala_message": error_message,
                "_adala_details": error_details,
            }
            return error_dct

        return response.data


class AsyncLiteLLMChatRuntime(LiteLLMInferenceSettings, AsyncRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion
    models to perform the skill. It uses async calls to OpenAI API.

    Attributes:
        inference_settings (LiteLLMInferenceSettings): Common inference settings for LiteLLM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    def as_inference_settings(self, **override_kwargs) -> LiteLLMInferenceSettings:
        inference_settings = LiteLLMInferenceSettings(
            **self.dict(include=LiteLLMInferenceSettings.model_fields.keys())
        )
        inference_settings.update(**override_kwargs)
        return inference_settings

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            messages = [{"role": "user", "content": "Hey, how's it going?"}]
            litellm.completion(
                messages=messages, **self.as_inference_settings(max_tokens=10).dict()
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
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        """Execute batch of requests with async calls to OpenAI API"""

        response_model = parse_template_to_pydantic_class(
            output_template, provided_field_schema=field_schema
        )

        extra_fields = extra_fields or {}
        user_prompts = batch.apply(
            lambda row: input_template.format(**row, **extra_fields), axis=1
        ).tolist()

        tasks = [
            asyncio.ensure_future(
                async_instructor_client.chat.completions.create(
                    messages=get_messages(
                        user_prompt, self.system_prompt, self.instruction_first
                    ),
                    response_model=response_model,
                    **self.as_inference_settings().dict(),
                )
            )
            for user_prompt in user_prompts
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # convert list of LLMResponse objects to the dataframe records
        df_data = []
        for response in responses:
            if isinstance(response, Exception):
                error_message = type(response).__name__
                # error_details = str(response)
                error_details = traceback.format_exc()
                if self.verbose:
                    print_error(error_message, error_details)
                # TODO change this format
                error_dct = {
                    "_adala_error": True,
                    "_adala_message": error_message,
                    "_adala_details": error_details,
                }
                df_data.append(error_dct)
            else:
                df_data.append(response.data)

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
            inference_settings=self.as_inference_settings(),
        )

        completion_text = completion.choices[0].message.content
        return {output_field_name: completion_text}
