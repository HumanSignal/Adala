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
    # _program = None

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

            # TODO: we can't initialize the program here because we don't have the template yet
            # self._program = guidance(self.llm_template, llm=self._llm)
        return self

    @classmethod
    def _ensure_correct_inputs(cls, inputs: List[str], input_prompt: str) -> Tuple[Dict, str]:
        """
        guidance use specific built-in keywords for inputs, e.g. {{text}}.
        We need to replace them with something else, e.g. {{text_}}.
        """
        corrected_inputs = {}
        fixed_prompt = input_prompt
        for input in inputs:
            if input == 'text':
                corrected_inputs['text'] = 'text_'
                # TODO: replace it with regex replace
                fixed_prompt = fixed_prompt.replace('{{text}}', '{{text_}}')
            else:
                corrected_inputs[input] = None
        return corrected_inputs, fixed_prompt

    def _process_record(self, record, program, instructions, inputs, outputs, extra_fields):
        verified_input = record.to_dict()
        if 'text' in verified_input:
            verified_input['text_'] = verified_input['text']
            del verified_input['text']
        verified_input.update(extra_fields)
        result = program(silent=True, instructions=instructions, **verified_input)
        verified_output = {field: result[field] for field in outputs}

        return verified_output

    def process_batch(
        self,
        batch: InternalDataFrame,
        prompt_template: str,
        instructions: str,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> InternalDataFrame:

        extra_fields = extra_fields or {}

        # extract inputs from prompt template
        # TODO: this is a very naive regex implementation - likely to fail in many cases
        # search for all occurrences of {{input}}
        inputs = re.findall(r'{{([^\}\s]+)}}', prompt_template)
        # inputs.remove('instructions')
        # search for all occurrences of {{...'output'...}}
        outputs = re.findall(r'\'(.*?)\'', prompt_template)

        corrected_inputs, fixed_prompt_template = self._ensure_correct_inputs(inputs, prompt_template)
        # TODO: it's not efficient way to initialize the program here - should be done once
        instructions_program = guidance(instructions, llm=self._llm)
        main_program = guidance(fixed_prompt_template, llm=self._llm)
        output = batch.progress_apply(
            self._process_record,
            axis=1,
            result_type='expand',
            program=main_program,
            instructions=instructions_program,
            inputs=corrected_inputs,
            outputs=outputs,
            extra_fields=extra_fields
        )
        return output


class CodeRuntime(Runtime):
    """
    Base class for code runtimes.
    """
