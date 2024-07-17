import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
import litellm
from adala.utils.internal_data import InternalDataFrame
from adala.utils.logs import print_error
from adala.utils.matching import match_options
from adala.utils.parse import parse_template, partial_str_format
from openai import NotFoundError
from pydantic import ConfigDict
from rich import print
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import AsyncRuntime, Runtime

logger = logging.getLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def async_create_completion(
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    instruction_first: bool = True,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Async version of create_completion function with error handling and session timeout.

    Args:
        model: model name. Refer to litellm supported models for how to pass
               this: https://litellm.vercel.app/docs/providers
        user_prompt: User prompt.
        system_prompt: System prompt.
        openai_api_key: OpenAI API key (if not set, will use OPENAI_API_KEY environment variable).
        instruction_first: Whether to put instructions first.
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.

    Returns:
        Dict[str, Any]: OpenAI response or error message.
    """
    messages = [{'role': 'user', 'content': user_prompt}]
    if system_prompt:
        if instruction_first:
            messages.insert(0, {'role': 'system', 'content': system_prompt})
        else:
            messages[0]['content'] += system_prompt

    try:
        completion = await litellm.acompletion(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        completion_text = completion.choices[0].message.content
        return {
            'text': completion_text,
            '_adala_error': False,
            '_adala_message': None,
            '_adala_details': None,
        }
    except Exception as e:
        # Handle other exceptions
        return {
            'text': None,
            '_adala_error': True,
            '_adala_message': type(e).__name__,
            '_adala_details': str(e),
        }


async def async_concurrent_create_completion(
    prompts: List[Dict],
    instruction_first: bool,
    model: str,
    max_tokens: int,
    temperature: float,
    api_key: Optional[str] = None,
):
    tasks = [
        asyncio.ensure_future(
            async_create_completion(
                user_prompt=prompt['user'],
                system_prompt=prompt['system'],
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                instruction_first=instruction_first,
            )
        )
        for prompt in prompts
    ]
    responses = await asyncio.gather(*tasks)
    return responses


class LiteLLMChatRuntime(Runtime):
    """
    Runtime that uses [LiteLLM API](https://litellm.vercel.app/docs) and chat completion models to perform the skill.

    Attributes:
        model: Model name, refer to LiteLLM's supported provider docs for
                    how to pass this for your model: https://litellm.vercel.app/docs/providers
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url: Can point to any implementation of your model's API.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # for @computed_field

    model: str
    api_key: Optional[str]
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 1000
    splitter: Optional[str] = None

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

    def execute(self, messages: List):
        """
        Execute LiteLLM request given list of messages in OpenAI API format
        """
        if self.verbose:
            print(f'LiteLLM request: {messages}')

        completion = litellm.completion(
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            max_tokens=self.max_tokens,
            # TODO temperature, **kwargs?
        )
        completion_text = completion.choices[0].message.content

        if self.verbose:
            print(f'LiteLLM response: {completion_text}')
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
        field_schema = field_schema or {}

        options = {}
        for field, schema in field_schema.items():
            if schema.get('type') == 'array':
                options[field] = schema.get('items', {}).get('enum', [])

        output_fields = parse_template(
            partial_str_format(output_template, **extra_fields),
            include_texts=True,
        )
        system_prompt = instructions_template
        user_prompt = input_template.format(**record, **extra_fields)
        messages = [{'role': 'system', 'content': system_prompt}]

        outputs = {}
        for output_field in output_fields:
            if output_field['type'] == 'text':
                if user_prompt is not None:
                    user_prompt += f"\n{output_field['text']}"
                else:
                    user_prompt = output_field['text']
            elif output_field['type'] == 'var':
                name = output_field['text']
                messages.append({'role': 'user', 'content': user_prompt})
                completion_text = self.execute(messages)
                if name in options:
                    completion_text = match_options(
                        completion_text, options[name]
                    )
                outputs[name] = completion_text
                messages.append(
                    {'role': 'assistant', 'content': completion_text}
                )
                user_prompt = None

        return outputs


class AsyncLiteLLMChatRuntime(AsyncRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion models to perform the skill.
    It uses async calls to OpenAI API.

    Attributes:
        model: OpenAI model name.
        openai_api_key: OpenAI API key. If not provided, will be taken from OPENAI_API_KEY environment variable.
        base_url: Can point to any implementation of the OpenAI API. Defaults to OpenAI's.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        temperature: Temperature for sampling, between 0 and 1. Higher values means the model will take more risks.
            Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            Defaults to 0.0.

        concurrent_clients: Number of concurrent clients to OpenAI API. More clients means more parallel requests, but
            also more money spent and more chances to hit the rate limit. Defaults to 10.
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

    def _init_http_client(self) -> httpx.AsyncClient:
        client = (
            httpx.AsyncClient(
                timeout=self.timeout,
            ),
        )
        litellm.aclient_session = client
        return client

    def init_runtime(self) -> 'Runtime':
        # check model availability
        try:
            if self.api_key:
                litellm.check_valid_key(model=self.model, api_key=self.api_key)
        except NotFoundError:
            raise ValueError(
                f'Requested model "{self.model}" is not available in your OpenAI account.'
            )
        self._http_client = self._init_http_client
        return self

    def _prepare_prompt(
        self,
        row,
        input_template: str,
        instructions_template: str,
        suffix: str,
        extra_fields: dict,
    ) -> Dict[str, str]:
        """Prepare input prompt for OpenAI API from the row of the dataframe"""
        return {
            'system': instructions_template,
            'user': input_template.format(**row, **extra_fields) + suffix,
        }

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

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        options = {}
        for field, schema in field_schema.items():
            if schema.get('type') == 'array':
                options[field] = schema.get('items', {}).get('enum', [])

        output_fields = parse_template(
            partial_str_format(output_template, **extra_fields),
            include_texts=True,
        )

        if len(output_fields) > 2:
            raise NotImplementedError('Only one output field is supported')

        suffix = ''
        outputs = []
        for output_field in output_fields:
            if output_field['type'] == 'text':
                suffix += output_field['text']

            elif output_field['type'] == 'var':
                name = output_field['text']
                # prepare prompts
                prompts = batch.apply(
                    lambda row: self._prepare_prompt(
                        row,
                        input_template,
                        instructions_template,
                        suffix,
                        extra_fields,
                    ),
                    axis=1,
                ).tolist()

                responses = await async_concurrent_create_completion(
                    prompts=prompts,
                    instruction_first=instructions_first,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    model=self.model,
                    api_key=self.api_key,
                )

                # parse responses, optionally match it with options
                for prompt, response in zip(prompts, responses):
                    completion_text = response.pop('text')
                    if self.verbose:
                        if response['error'] is not None:
                            print_error(
                                f'Prompt: {prompt}\nLiteLLM API error: {response}'
                            )
                        else:
                            print(
                                f'Prompt: {prompt}\nLiteLLM API response: {completion_text}'
                            )
                    if name in options and completion_text is not None:
                        completion_text = match_options(
                            completion_text, options[name]
                        )
                    # still technically possible to have a name collision here with the error, message, details fields
                    # `name in options` is only `True` for categorical variables, but is never `True` for freeform text generation
                    response[name] = completion_text
                    outputs.append(response)

        # TODO: note that this doesn't work for multiple output fields e.g. `Output {output1} and Output {output2}`
        output_df = InternalDataFrame(outputs)
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
        Execute LiteLLM request given record and templates for input, instructions and output.

        Args:
            record: Record to be used for input, instructions and output templates.
            input_template: Template for input message.
            instructions_template: Template for instructions message.
            output_template: Template for output message.
            extra_fields: Extra fields to be used in templates.
            field_schema: Field jsonschema to be used for parsing templates.
                         Field schema must contain "format": "uri" for image fields. For example:
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
