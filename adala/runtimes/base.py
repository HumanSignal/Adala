import enum
import guidance
import re

from tqdm import tqdm
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional, Tuple, Any
from adala.datasets.base import InternalDataFrame

tqdm.pandas()


class Runtime(BaseModel, ABC):
    """
    Base class for runtimes.
    """
    verbose: bool = False

    @model_validator(mode='after')
    def init_runtime(self):
        """
        Check that runtime is valid.
        Use this method to initialize runtime.
        """
        return self


class LLMRuntimeModelType(enum.Enum):
    OpenAI = 'OpenAI'
    Transformers = 'Transformers'


class LLMRuntime(Runtime):
    """
    Base class for LLM runtimes.
    """

    llm_runtime_type: LLMRuntimeModelType = LLMRuntimeModelType.OpenAI
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

    class Config:
        arbitrary_types_allowed = True

    def init_runtime(self):
        if not self._llm:

            # create an LLM instance
            if self.llm_runtime_type.value == LLMRuntimeModelType.OpenAI.value:
                self._llm = guidance.llms.OpenAI(**self.llm_params)
            elif self.llm_runtime_type.value == LLMRuntimeModelType.Transformers.value:
                self._llm = guidance.llms.Transformers(**self.llm_params)
            else:
                raise NotImplementedError(f'LLM runtime type {self.llm_runtime_type} is not implemented.')

            self._program = guidance(self._llm_template, llm=self._llm)
        return self

    def get_outputs(self, output_template: str) -> List[str]:
        # search for all occurrences of {{...'output'...}}
        # TODO: this is a very naive regex implementation - likely to fail in many cases
        outputs = re.findall(r'\'(.*?)\'', output_template)
        return outputs

    def _process_record(
        self,
        record,
        program,
        extra_fields,
        outputs=None
    ):
        if not isinstance(record, dict):
            record = record.to_dict()
        else:
            record = record.copy()
        verified_input = record
        # exclude guidance parameter from input
        if 'text' in verified_input:
            verified_input['text_'] = verified_input['text']
            del verified_input['text']

        verified_input.update(extra_fields)
        result = program(
            # silent=not self.verbose,
            **verified_input
        )
        if outputs is None:
            verified_output = {'': str(result)}
        else:
            verified_output = {field: result[field] for field in outputs}

        return verified_output

    def get_input_program(self, input_template):
        # fix input template in case "text" is presented there - there might be other paramater names as well...
        fixed_input_template = input_template
        if '{{text}}' in fixed_input_template:
            fixed_input_template = fixed_input_template.replace('{{text}}', '{{text_}}')
        input_program = guidance(fixed_input_template, llm=self._llm)
        return input_program

    def get_output_program(self, output_template):
        return guidance(output_template, llm=self._llm)

    def get_instructions_program(self, instructions):
         return guidance(instructions, llm=self._llm)

    def process_record(
        self,
        record: Dict[str, Any],
        input_template: str,
        output_template: str,
        instructions: str,
    ) -> Dict[str, Any]:

        outputs = re.findall(r'\'(.*?)\'', output_template)

        input = record.copy()
        input.update({
            'input_program': self.get_input_program(input_template),
            'output_program': self.get_output_program(output_template),
            'instructions_program': self.get_instructions_program(instructions),
        })
        output = self._process_record(
            record=input,
            program=self._program,
            outputs=outputs,
            extra_fields={}
        )
        return output

    def process_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        output_template: str,
        instructions: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> InternalDataFrame:

        outputs = self.get_outputs(output_template)

        extra_fields = extra_fields or {}
        # copy extra fields to avoid modification of the original dict
        extra_fields = extra_fields.copy()
        # TODO: it's not efficient way to initialize the program here - should be done once
        extra_fields.update({
            'input_program': self.get_input_program(input_template),
            'output_program': self.get_output_program(output_template),
            'instructions_program': self.get_instructions_program(instructions),
        })
        output = batch.progress_apply(
            self._process_record,
            axis=1,
            result_type='expand',
            program=self._program,
            outputs=outputs,
            extra_fields=extra_fields
        )
        return output

    def process_batch_inputs(
        self,
        batch: InternalDataFrame,
        input_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> InternalDataFrame:

        output = batch.progress_apply(
            self._process_record,
            axis=1,
            result_type='expand',
            program=self.get_input_program(input_template),
            extra_fields=extra_fields or {}
        )
        return output


class CodeRuntime(Runtime):
    """
    Base class for code runtimes.
    """
