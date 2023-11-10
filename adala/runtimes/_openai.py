import os
from rich import print
from openai import OpenAI, NotFoundError
from pydantic import model_validator, field_validator, ValidationInfo, Field
from typing import Optional, Dict, Any, List
from .base import (
    LLMRuntime, LLMRuntimeType, LLMRuntimeModelType, Runtime
)
from adala.utils.logs import print_error
from adala.utils.internal_data import InternalDataFrame
from adala.utils.parse import parse_template, partial_str_format


class OpenAIRuntime(LLMRuntime):
    """Runtime class specifically designed for OpenAI models.

    This class is tailored to use OpenAI models, particularly GPT models.
    It inherits from the `LLMRuntime` class and thus can utilize its functionalities but specializes 
    for the OpenAI ecosystem.

    Attributes:
        api_key (str): The API key required to access OpenAI's API.
        gpt_model_name (str): Name of the GPT model. Defaults to 'gpt-3.5-turbo-instruct'.
        temperature (float): Sampling temperature for the GPT model's output. 
                             A higher value makes output more random, while a lower value makes it more deterministic.
                             Defaults to 0.0.
    """

    api_key: Optional[str] = None
    gpt_model_name: Optional[str] = Field(default='gpt-3.5-turbo-instruct', alias='model')
    temperature: Optional[float] = 0.0

    def _check_api_key(self):
        if self.api_key:
            return
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print_error(
                'OpenAI API key is not provided. Please set the OPENAI_API_KEY environment variable:\n\n'
                'export OPENAI_API_KEY=your-openai-api-key\n\n'
                'or set the `api_key` attribute of the `OpenAIRuntime` python class:\n\n'
                f'{self.__class__.__name__}(..., api_key="your-openai-api-key")\n\n'
                f'Read more about OpenAI API keys at https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key')
            raise ValueError('OpenAI API key is not provided.')

    def _check_model_availability(self):
        models = openai.Model.list(api_key=self.api_key)
        models = set(model['id'] for model in models['data'])
        # models = openai.models.list()
        # models = set(model.id for model in models.data)
        if self.gpt_model_name not in models:
            print_error(
                f'Requested model "{self.gpt_model_name}" is not available in your OpenAI account. '
                f'Available models are: {models}\n\n'
                f'Try to change the runtime settings for {self.__class__.__name__}, for example:\n\n'
                f'{self.__class__.__name__}(..., model="gpt-3.5-turbo")\n\n'
            )
            raise ValueError(f'Requested model {self.gpt_model_name} is not available in your OpenAI account.')

    def init_runtime(self):
        self._check_api_key()
        self._check_model_availability()

        student_models = {'gpt-3.5-turbo-instruct', 'text-davinci-003'}
        teacher_models = {'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4-1106-preview', 'gpt-4-vision-preview'}

        if self.gpt_model_name in student_models:
            self.llm_runtime_type = LLMRuntimeType.STUDENT
        elif self.gpt_model_name in teacher_models:
            self.llm_runtime_type = LLMRuntimeType.TEACHER
        else:
            raise NotImplementedError(f'Not supported model: {self.gpt_model_name}.')

        self.llm_runtime_model_type = LLMRuntimeModelType.OpenAI
        self.llm_params = {
            'model': self.gpt_model_name,
            'temperature': self.temperature,
            'api_key': self.api_key
        }
        self._create_program()
        return self


class OpenAIChatRuntime(Runtime):
    openai_model: str = Field(alias='model')
    openai_api_key: Optional[str] = Field(default=os.getenv('OPENAI_API_KEY'), alias='api_key')
    max_tokens: Optional[int] = 1000

    _client: OpenAI = None

    def init_runtime(self) -> 'Runtime':
        if self._client is None:
            self._client = OpenAI(api_key=self.openai_api_key)

        # check model availability
        try:
            self._client.models.retrieve(self.openai_model)
        except NotFoundError:
            raise ValueError(f'Requested model "{self.openai_model}" is not available in your OpenAI account.')
        return self

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
    ) -> Dict[str, str]:

        extra_fields = extra_fields or {}

        output_fields = parse_template(partial_str_format(output_template, **extra_fields), include_texts=False)
        if len(output_fields) > 1:
            raise NotImplementedError(f'{self.__class__.__name__} does not support multiple output fields. '
                                      f'Found: {output_fields}')
        output_field = output_fields[0]
        output_field_name = output_field['text']
        system_prompt = instructions_template.format(**record, **extra_fields)
        user_prompt = input_template.format(**record, **extra_fields)
        # TODO: this truncates the suffix of the output template
        # for example, output template "Output: {answer} is correct" results in output_prefix "Output: "
        output_prefix = output_template[:output_field['start']]
        user_prompt += f'\n\n{output_prefix}'

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if self.verbose:
            print(f'OpenAI request: {messages}')

        completion = self._client.chat.completions.create(
            model=self.openai_model,
            messages=messages
        )
        completion_text = completion.choices[0].message.content
        return {output_field_name: completion_text}


class OpenAIVisionRuntime(OpenAIChatRuntime):

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
    ) -> Dict[str, str]:

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        output_fields = parse_template(partial_str_format(output_template, **extra_fields), include_texts=False)

        if len(output_fields) > 1:
            raise NotImplementedError(f'{self.__class__.__name__} does not support multiple output fields. '
                                      f'Found: {output_fields}')
        output_field = output_fields[0]
        output_field_name = output_field['text']

        input_fields = parse_template(input_template)

        # split input template into text and image parts
        input_text = ''
        content = [{
            'type': 'text',
            'text': instructions_template.format(**dict(**record, **extra_fields))
        }]
        for field in input_fields:
            if field['type'] == 'text':
                input_text += field['text']
            elif field['type'] == 'var':
                if field['text'] not in field_schema:
                    input_text += record[field['text']]
                elif field_schema[field['text']]['type'] == 'string':
                    if field_schema[field['text']].get('format') == 'uri':
                        if input_text:
                            content.append({'type': 'text', 'text': input_text})
                            input_text = ''
                        content.append({'type': 'image_url', 'image_url': record[field['text']]})
                    else:
                        input_text += record[field['text']]
                else:
                    raise ValueError(f'Unsupported field type: {field_schema[field["text"]]["type"]}')
        if input_text:
            content.append({'type': 'text', 'text': input_text})

        if self.verbose:
            print(f'**Prompt content**:\n{content}')

        completion = self._client.chat.completions.create(
            model=self.openai_model,
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=self.max_tokens
        )

        completion_text = completion.choices[0].message.content
        return {output_field_name: completion_text}
