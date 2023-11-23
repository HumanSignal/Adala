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
    """
    Runtime for LLMs powered by [guidance](https://github.com/guidance-ai/guidance).
    Here you can rely on constrained generation problem formulation,
    which require structured programs to be defined on the input and output data.
    """

    llm_runtime_model_type: GuidanceModelType = GuidanceModelType.OpenAI
    llm_params: Dict[str, str] = {
        'model': 'gpt-3.5-turbo-instruct',
        # 'max_tokens': 10,
        'temperature': 0,
    }

    _llm = None
    # _program = None
    #     # do not override this template
    #     _llm_template: str = '''\
    # {{>instructions_program}}
    #
    # {{>input_program}}
    # {{>output_program}}'''

    # do not override this template
    _llm_templates: Dict[str, str] = {
        True: '''\
{{>instructions_program}}
{{>input_program}}
{{>output_program}}''',
        False: '''\
{{>input_program}}
{{>instructions_program}}
{{>output_program}}'''
    }

    def init_runtime(self) -> Runtime:
        """
        Initializes the runtime.
        """

        # create an LLM instance
        if self.llm_runtime_model_type.value == GuidanceModelType.OpenAI.value:
            self._llm = guidance.llms.OpenAI(**self.llm_params)
        elif self.llm_runtime_model_type.value == GuidanceModelType.Transformers.value:
            self._llm = guidance.llms.Transformers(**self.llm_params)
        else:
            raise NotImplementedError(f'LLM runtime type {self.llm_runtime_model_type} is not implemented.')
        # self._program = guidance(self._llm_templates[self.instruction_first], llm=self._llm, silent=not self.verbose)
        return self

    def _input_template_to_guidance(self, input_template, program_input):
        # TODO: this check is brittle, will likely to fail in various cases
        # exclude guidance parameter from input
        if 'text' in program_input:
            program_input['text_'] = program_input['text']
            del program_input['text']
        if '{text}' in input_template:
            input_template = input_template.replace('{text}', '{text_}')

        fields = parse_template(input_template, include_texts=False)
        # replace {field_name} with {{field_name}}
        for input_field in fields:
            field_name = input_field['text']
            if field_name in program_input:
                input_template = input_template.replace(f'{{{field_name}}}', f'{{{{{field_name}}}}}')
        return input_template

    def _output_template_to_guidance(self, output_template, program_input, output_fields, field_schema):
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
                program_input[f'{field_name}_options'] = field_schema[field_name]['items']['enum']
                output_template = output_template.replace(f'{{{field_name}}}',
                                                          f'{{{{select \'{field_name}\' options={field_name}_options}}}}')
            else:
                # In simple generation scenario, replace {field_name} with {{gen 'field_name'}}
                output_template = output_template.replace(f'{{{field_name}}}', f'{{{{gen \'{field_name}\'}}}}')
        return output_template

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> Dict[str, str]:
        """
        Generates a record from a record.

        Args:
            record (Dict[str, str]): Input record.
            input_template (str): Input template.
            instructions_template (str): Instructions template.
            output_template (str): Output template.
            extra_fields (Optional[Dict[str, Any]], optional): Extra fields. Defaults to None.
            field_schema (Optional[Dict], optional): Field schema. Defaults to None.
            instructions_first (bool, optional): Whether to put instructions first. Defaults to True.

        Returns:
            Dict[str, str]: Generated record.
        """

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        if not isinstance(record, dict):
            record = record.to_dict()
        else:
            record = record.copy()
        program_input = record
        program_input.update(extra_fields)

        output_fields = parse_template(partial_str_format(output_template, **extra_fields), include_texts=False)

        input_template = self._input_template_to_guidance(input_template, program_input)
        instructions_template = self._input_template_to_guidance(instructions_template, program_input)
        output_template = self._output_template_to_guidance(output_template, program_input, output_fields, field_schema)

        program_input['input_program'] = guidance(input_template, llm=self._llm)
        program_input['instructions_program'] = guidance(instructions_template, llm=self._llm)
        program_input['output_program'] = guidance(output_template, llm=self._llm)

        if self.verbose:
            print(program_input)

        program = guidance(self._llm_templates[instructions_first], llm=self._llm, silent=not self.verbose)

        result = program(
            silent=not self.verbose,
            **program_input
        )

        output = {}
        for output_field in output_fields:
            if output_field['text'] in extra_fields:
                continue
            output[output_field['text']] = result[output_field['text']]
        return output
