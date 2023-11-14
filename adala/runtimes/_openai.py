import os
from rich import print

from typing import Optional, Dict, Any, List

def check_if_new_openai_version():
    # check openai package version
    from openai import __version__ as openai_version
    from packaging import version
    if version.parse(openai_version) >= version.parse('1.0.0'):
        return True
    else:
        return False

# if version is higher than 1.0.0, then import OpenAI class
if check_if_new_openai_version():
    from openai import OpenAI, NotFoundError
# otherwise, use old style API
else:
    import openai
    OpenAI = Any

from pydantic import model_validator, field_validator, ValidationInfo, Field
from .base import Runtime
from adala.utils.logs import print_error
from adala.utils.internal_data import InternalDataFrame
from adala.utils.parse import parse_template, partial_str_format


class OpenAIChatRuntime(Runtime):
    openai_model: str = Field(alias='model')
    openai_api_key: Optional[str] = Field(default=os.getenv('OPENAI_API_KEY'), alias='api_key')
    max_tokens: Optional[int] = 1000

    _client: OpenAI = None

    def init_runtime(self) -> 'Runtime':
        # check openai package version
        if check_if_new_openai_version():

            if self._client is None:
                self._client = OpenAI(api_key=self.openai_api_key)

            # check model availability
            try:
                self._client.models.retrieve(self.openai_model)
            except NotFoundError:
                raise ValueError(f'Requested model "{self.openai_model}" is not available in your OpenAI account.')
        else:
            # deprecated
            models = openai.Model.list(api_key=self.openai_api_key)
            models = set(model['id'] for model in models['data'])
            if self.openai_model not in models:
                print_error(
                    f'Requested model "{self.openai_model}" is not available in your OpenAI account. '
                    f'Available models are: {models}\n\n'
                    f'Try to change the runtime settings for {self.__class__.__name__}, for example:\n\n'
                    f'{self.__class__.__name__}(..., model="gpt-3.5-turbo")\n\n'
                )
                raise ValueError(f'Requested model {self.openai_model} is not available in your OpenAI account.')
        return self

    def execute(self, messages: List):
        if self.verbose:
            print(f'OpenAI request: {messages}')

        if check_if_new_openai_version():
            completion = self._client.chat.completions.create(
                model=self.openai_model,
                messages=messages
            )
            completion_text = completion.choices[0].message.content
        else:
            # deprecated
            completion = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=messages
            )
            completion_text = completion.choices[0]['message']['content']
        return completion_text

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

        completion_text = self.execute(messages)
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

        if not check_if_new_openai_version():
            raise NotImplementedError(f'{self.__class__.__name__} requires OpenAI API version 1.0.0 or higher.')

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
