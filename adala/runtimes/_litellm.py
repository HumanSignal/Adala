import logging
from typing import Any, Dict, List, Optional

import litellm
from adala.utils.internal_data import InternalDataFrame
from adala.utils.logs import print_error
from adala.utils.matching import match_options
from adala.utils.parse import parse_template, partial_str_format, parse_template_to_pydantic_class
from adala.utils.llm import parallel_async_get_llm_response, get_llm_response
from openai import NotFoundError
from pydantic import ConfigDict, field_validator
from rich import print

from .base import AsyncRuntime, Runtime
from ..utils.llm import parallel_async_get_llm_response

logger = logging.getLogger(__name__)


class LiteLLMChatRuntime(Runtime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and chat
    completion models to perform the skill.

    Attributes:
        model: Model name, refer to LiteLLM's supported provider docs for
               how to pass this for your model: https://litellm.vercel.app/docs/providers
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url: Points to the endpoint where your model is hosted
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        splitter: Splitter to use for splitting messages. Defaults to None.
        temperature: Temperature for sampling, between 0 and 1.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # for @computed_field

    model: str
    api_key: Optional[str]
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 1000
    splitter: Optional[str] = None
    temperature: Optional[float] = 0.0

    def init_runtime(self) -> 'Runtime':
        # check model availability
        try:
            if self.api_key:
                litellm.check_valid_key(model=self.model, api_key=self.api_key)
        except NotFoundError:
            raise ValueError(
                f'Requested model "{self.model}" is not available with your api_key.'
            )
        return self

    def get_llm_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        response = get_llm_response(
            messages=messages,
            model=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response['data']['text']

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
            output_template,
            provided_field_schema=field_schema
        )

        response = get_llm_response(
            user_prompt=input_template.format(**record, **extra_fields),
            system_prompt=instructions_template,
            instruction_first=instructions_first,
            model=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_model=response_model
        )

        response.update(**response.get('data', {}))
        response.pop('data', None)
        return response


class AsyncLiteLLMChatRuntime(AsyncRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion
    models to perform the skill. It uses async calls to OpenAI API.

    Attributes:
        model: OpenAI model name.
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url: Points to the endpoint where your model is hosted
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        temperature: Temperature for sampling, between 0 and 1. Higher values
                     means the model will take more risks. Try 0.9 for more
                     creative applications, and 0 (argmax sampling) for ones
                     with a well-defined answer.
                     Defaults to 0.0.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # for @computed_field

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
    splitter: Optional[str] = None
    timeout: Optional[int] = 10

    @field_validator("concurrency", mode="before")
    def check_concurrency(cls, value) -> int:
        value = value or -1
        if value < 1:
            raise NotImplementedError(
                "You must explicitly specify the number of concurrent clients for AsyncOpenAIChatRuntime. "
                "Set `AsyncOpenAIChatRuntime(concurrency=10, ...)` or any other positive integer. ")
        return value

    def init_runtime(self) -> 'Runtime':
        # check model availability
        try:
            if self.api_key:
                litellm.check_valid_key(model=self.model, api_key=self.api_key)
        except NotFoundError:
            raise ValueError(
                f'Requested model "{self.model}" is not available in your OpenAI account.'
            )
        return self

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
            output_template,
            provided_field_schema=field_schema
        )

        extra_fields = extra_fields or {}
        user_prompts = batch.apply(lambda row: input_template.format(**row, **extra_fields), axis=1).tolist()

        responses = await parallel_async_get_llm_response(
            user_prompts=user_prompts,
            system_prompt=instructions_template,
            instruction_first=instructions_first,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            model=self.model,
            api_key=self.api_key,
            timeout=self.timeout,
            response_model=response_model
        )

        for response in responses:
            response.update(**response.get('data', {}))
            response.pop('data', None)

        output_df = InternalDataFrame(responses)
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
        raise NotImplementedError('record_to_record is not implemented')


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
                f'{self.__class__.__name__} does not support multiple output fields. '
                f'Found: {output_fields}'
            )
        output_field = output_fields[0]
        output_field_name = output_field['text']

        input_fields = parse_template(input_template)

        # split input template into text and image parts
        input_text = ''
        content = [
            {
                'type': 'text',
                'text': instructions_template,
            }
        ]
        for field in input_fields:
            if field['type'] == 'text':
                input_text += field['text']
            elif field['type'] == 'var':
                if field['text'] not in field_schema:
                    input_text += record[field['text']]
                elif field_schema[field['text']]['type'] == 'string':
                    if field_schema[field['text']].get('format') == 'uri':
                        if input_text:
                            content.append(
                                {'type': 'text', 'text': input_text}
                            )
                            input_text = ''
                        content.append(
                            {
                                'type': 'image_url',
                                'image_url': record[field['text']],
                            }
                        )
                    else:
                        input_text += record[field['text']]
                else:
                    raise ValueError(
                        f'Unsupported field type: {field_schema[field["text"]]["type"]}'
                    )
        if input_text:
            content.append({'type': 'text', 'text': input_text})

        if self.verbose:
            print(f'**Prompt content**:\n{content}')

        completion = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            messages=[{'role': 'user', 'content': content}],
            max_tokens=self.max_tokens,
        )

        completion_text = completion.choices[0].message.content
        return {output_field_name: completion_text}
