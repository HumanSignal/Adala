import logging
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm.exceptions import AuthenticationError
from adala.utils.internal_data import InternalDataFrame
from adala.utils.logs import print_error
from adala.utils.parse import (
    parse_template,
    partial_str_format,
    parse_template_to_pydantic_class,
)
from adala.utils.llm import (
    parallel_async_get_llm_response,
    get_llm_response,
    ConstrainedLLMResponse,
    UnconstrainedLLMResponse,
    ErrorLLMResponse,
    LiteLLMInferenceSettings,
)
from pydantic import ConfigDict, field_validator
from rich import print

from .base import AsyncRuntime, Runtime

logger = logging.getLogger(__name__)


class LiteLLMChatRuntime(LiteLLMInferenceSettings, Runtime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and chat
    completion models to perform the skill.

    Attributes:
        inference_settings (LiteLLMInferenceSettings): Common inference settings for LiteLLM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    def as_inference_settings(self, **override_kwargs) -> LiteLLMInferenceSettings:
        inference_settings = LiteLLMInferenceSettings(**self.dict(include=LiteLLMInferenceSettings.model_fields.keys()))
        inference_settings.update(**override_kwargs)
        return inference_settings

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            messages = [{"role": "user", "content": "Hey, how's it going?"}]
            get_llm_response(
                messages=messages,
                inference_settings=self.as_inference_settings(max_tokens=10)
            )
        except AuthenticationError:
            raise ValueError(
                f'Requested model "{self.model}" is not available with your api_key and settings.'
            )
        except Exception as e:
            raise ValueError(f'Failed to check availability of requested model "{self.model}": {e}')
        return self

    def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        # TODO: sunset this method in favor of record_to_record
        if self.verbose:
            print(f"**Prompt content**:\n{messages}")
        response: Union[ErrorLLMResponse, UnconstrainedLLMResponse] = get_llm_response(
            messages=messages,
            inference_settings=self.as_inference_settings(),
        )
        if isinstance(response, ErrorLLMResponse):
            raise ValueError(f"{response.adala_message}\n{response.adala_details}")
        if self.verbose:
            print(f"**Response**:\n{response.text}")
        return response.text

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

        response: Union[ConstrainedLLMResponse, ErrorLLMResponse] = get_llm_response(
            user_prompt=input_template.format(**record, **extra_fields),
            system_prompt=instructions_template,
            instruction_first=instructions_first,
            response_model=response_model,
            inference_settings=self.as_inference_settings(),
        )

        if isinstance(response, ErrorLLMResponse):
            if self.verbose:
                print_error(response.adala_message, response.adala_details)
            return response.model_dump(by_alias=True)

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
        inference_settings = LiteLLMInferenceSettings(**self.dict(include=LiteLLMInferenceSettings.model_fields.keys()))
        inference_settings.update(**override_kwargs)
        return inference_settings

    def init_runtime(self) -> "Runtime":
        # check model availability
        # extension of litellm.check_valid_key for non-openai deployments
        try:
            messages = [{"role": "user", "content": "Hey, how's it going?"}]
            get_llm_response(
                messages=messages,
                inference_settings=self.as_inference_settings(max_tokens=10)
            )
        except AuthenticationError:
            raise ValueError(
                f'Requested model "{self.model}" is not available with your api_key and settings.'
            )
        except Exception as e:
            raise ValueError(f'Failed to check availability of requested model "{self.model}": {e}')
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

        responses: List[Union[ConstrainedLLMResponse, ErrorLLMResponse]] = (
            await parallel_async_get_llm_response(
                user_prompts=user_prompts,
                system_prompt=instructions_template,
                instruction_first=instructions_first,
                response_model=response_model,
                inference_settings=self.as_inference_settings(),
            )
        )

        # convert list of LLMResponse objects to the dataframe records
        df_data = []
        for response in responses:
            if isinstance(response, ErrorLLMResponse):
                if self.verbose:
                    print_error(response.adala_message, response.adala_details)
                df_data.append(response.model_dump(by_alias=True))
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
