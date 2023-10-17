import enum
import guidance
import re

from tqdm import tqdm
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, Dict


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

    class Config:
        arbitrary_types_allowed = True

    def init_runtime(self):
        if not self._program:

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
    def _extract_output_names(cls, text: str) -> List[str]:
        """
        Extract names of output fields from the template.
        """
        # Search for all occurrences of {{...}}
        matches = re.findall(r'{{(.*?)}}', text)

        names = []

        for match in matches:
            # Extract strings enclosed in single or double quotes within the match
            quoted_strings = re.findall(r'\'(.*?)\'|"(.*?)"', match)

            for s in quoted_strings:
                # Add non-empty matches to the names list
                if s[0]:
                    names.append(s[0])
                elif s[1]:
                    names.append(s[1])
        return names

    def process_batch(self, batch: List[Dict], prompt_template: str, extra_fields: Dict) -> List[Dict]:
        output = []
        output_names = self._extract_output_names(prompt_template)
        if 'text' in output_names or 'text' in extra_fields:
            raise ValueError('The field with name "text" is not allowed.')
        # TODO: it's not efficient way to initialize the program here - should be done once
        self._program = guidance(prompt_template, llm=self._llm)
        for record in tqdm(batch, disable=not self.verbose, desc='Processing batch'):
            if 'text' in record:
                raise ValueError('The field with name "text" is not allowed.')
            input = {**record, **extra_fields}
            result = self._program(silent=True, **input)
            output.append({name: result[name] for name in output_names})
        return output


class CodeRuntime(Runtime):
    """
    Base class for code runtimes.
    """
