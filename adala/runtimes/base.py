import enum
import guidance
import re

from tqdm import tqdm
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional, Tuple
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

    def _process_record(self, record, program, inputs, outputs, extra_fields):
        input = extra_fields.copy()
        for orig_name, replacement in inputs.items():
            if orig_name in input:
                continue
            input[replacement or orig_name] = record[orig_name]

        result = program(silent=True, **input)

        output = {field: result[field] for field in outputs}

        return output

    def process_batch(
        self,
        batch: InternalDataFrame,
        prompt_template: str,
        inputs: List[str],
        outputs: List[str],
        extra_fields: Optional[Dict] = None
    ) -> InternalDataFrame:

        extra_fields = extra_fields or {}
        corrected_inputs, fixed_prompt_template = self._ensure_correct_inputs(inputs, prompt_template)
        # TODO: it's not efficient way to initialize the program here - should be done once
        self._program = guidance(fixed_prompt_template, llm=self._llm)
        output = batch.progress_apply(
            self._process_record,
            axis=1,
            result_type='expand',
            program=self._program,
            inputs=corrected_inputs,
            outputs=outputs,
            extra_fields=extra_fields
        )
        return output


class CodeRuntime(Runtime):
    """
    Base class for code runtimes.
    """
