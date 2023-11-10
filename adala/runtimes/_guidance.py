import guidance
import enum
import re
from rich import print
from typing import Dict, Optional, Any
from .base import Runtime
from adala.utils.parse import parse_template, partial_str_format


class GuidanceModelType(enum.Enum):
    """Enumeration for LLM runtime model types."""
    OpenAI = 'OpenAI'
    Transformers = 'Transformers'


class GuidanceRuntime(Runtime):
    llm_runtime_model_type: GuidanceModelType = GuidanceModelType.OpenAI
    llm_params: Dict[str, str] = {
        'model': 'gpt-3.5-turbo-instruct',
        # 'max_tokens': 10,
        # 'temperature': 0,
    }

    _llm = None
    _program = None
    # do not override this template
    _llm_template: str = '''\
{{>instructions_program}}

{{>input_program}}
{{>output_program}}'''

    def init_runtime(self) -> Runtime:
        # create an LLM instance
        if self.llm_runtime_model_type.value == GuidanceModelType.OpenAI.value:
            self._llm = guidance.llms.OpenAI(**self.llm_params)
        elif self.llm_runtime_model_type.value == GuidanceModelType.Transformers.value:
            self._llm = guidance.llms.Transformers(**self.llm_params)
        else:
            raise NotImplementedError(f'LLM runtime type {self.llm_runtime_model_type} is not implemented.')
        self._program = guidance(self._llm_template, llm=self._llm, silent=not self.verbose)
        return self

    def _double_brackets(self, text):
        # This regex replaces occurrences of {word} with {{word}},
        # but ignores occurrences of {{word}}.
        # Negative lookbehind (?<!\{) ensures that the { is not preceded by another {
        # Negative lookahead (?!}) ensures that the } is not followed by another }
        return re.sub(r'(?<!\{)\{(\w+)\}(?!})', r'{{\1}}', text)

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
    ) -> Dict[str, str]:

        extra_fields = extra_fields or {}
        field_types = field_schema or {}

        if not isinstance(record, dict):
            record = record.to_dict()
        else:
            record = record.copy()
        program_input = record

        output_fields = parse_template(partial_str_format(output_template, **record), include_texts=False)
        for output_field in output_fields:
            field_name = output_field['text']
            if field_name in field_schema and field_schema[field_name]['type'] == 'array':
                # when runtime is called with a categorical field:
                #    runtime.record_to_record(
                #        ...,
                #        output_template='Predictions: {labels}',
                #        field_schema={'labels': {'type': 'array', 'items': {'type': 'string', 'enum': ['a', 'b', 'c']}}}
                #    )
                # replace {field_name} with {select 'field_name' options=field_name_options}
                # and add "field_name_options" to program input
                output_template = output_template.replace(
                    f'{{{field_name}}}',
                    f'{{{{select \'{field_name}\' options={field_name}}}}}'
                )
                program_input[field_name] = field_types[field_name]['items']['enum']

        # exclude guidance parameter from input
        if 'text' in program_input:
            program_input['text_'] = program_input['text']
            del program_input['text']
            # TODO: this check is brittle, will likely to fail in various cases
            if '{text}' in input_template:
                input_template = input_template.replace('{text}', '{text_}')
        program_input['input_program'] = guidance(self._double_brackets(input_template), llm=self._llm, silent=not self.verbose)
        program_input['output_program'] = guidance(self._double_brackets(output_template), llm=self._llm)
        program_input['instructions_program'] = guidance(self._double_brackets(instructions_template), llm=self._llm)
        program_input.update(extra_fields)

        if self.verbose:
            print(program_input)

        result = self._program(
            silent=not self.verbose,
            **program_input
        )

        output = {}
        for output_field in output_fields:
            if output_field['text'] in extra_fields:
                continue
            output[output_field['text']] = result[output_field['text']]
        return output
