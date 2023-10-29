import enum
import guidance
import re

from tqdm import tqdm
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional, Tuple, Any
from adala.datasets.base import InternalDataFrame
from adala.utils.logs import print_text

tqdm.pandas()


class Runtime(BaseModel, ABC):
    """
    Base class representing a generic runtime environment.

    Attributes:
        verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
    """
    verbose: bool = False

    @model_validator(mode='after')
    def init_runtime(self):
        """Initializes the runtime.

        This method should be used to validate and potentially initialize the runtime instance.

        Returns:
            Runtime: The initialized runtime instance.
        """  
        return self


class LLMRuntimeType(enum.Enum):
    STUDENT = 'student'
    TEACHER = 'teacher'


class LLMRuntimeModelType(enum.Enum):
    """Enumeration for LLM runtime model types."""    
    OpenAI = 'OpenAI'
    Transformers = 'Transformers'


class LLMRuntime(Runtime):
    """
    Class representing an LLM runtime environment.

    Attributes:
        llm_runtime_type (LLMRuntimeModelType): Type of the LLM runtime. Defaults to OpenAI.
        llm_params (Dict[str, str]): Parameters for the LLM runtime. Defaults to a basic GPT-3.5 configuration.
    
        _llm: Internal instance for the LLM model. Initialized in `init_runtime`.
        _program: Program instance used for guidance. Initialized in `init_runtime`.
        _llm_template (str): Template string for LLM guidance.
    """
    llm_runtime_type: LLMRuntimeType = LLMRuntimeType.STUDENT
    llm_runtime_model_type: LLMRuntimeModelType = LLMRuntimeModelType.OpenAI
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

    def _create_program(self):
        # create an LLM instance
        if self.llm_runtime_model_type.value == LLMRuntimeModelType.OpenAI.value:
            self._llm = guidance.llms.OpenAI(**self.llm_params)
        elif self.llm_runtime_model_type.value == LLMRuntimeModelType.Transformers.value:
            self._llm = guidance.llms.Transformers(**self.llm_params)
        else:
            raise NotImplementedError(f'LLM runtime type {self.llm_runtime_model_type} is not implemented.')
        self._program = guidance(self._llm_template, llm=self._llm, silent=not self.verbose)

    def init_runtime(self):
        """Initializes the LLM runtime environment.

        Creates an LLM instance based on the runtime type and parameters.

        Returns:
            LLMRuntime: Initialized runtime instance.
        """
        self._create_program()
        return self

    def get_outputs(self, output_template: Optional[str] = None) -> List[str]:
        """Extracts output fields from the output template.

        Args:
            output_template (str): The template string to extract output fields from.

        Returns:
            List[str]: List of extracted output fields.
        """
        # search for all occurrences of {{...'output'...}}
        # TODO: this is a very naive regex implementation - likely to fail in many cases
        if output_template is None:
            return []
        outputs = re.findall(r'\'(.*?)\'', output_template)
        return outputs

    def _process_record(
        self,
        record,
        program,
        extra_fields,
        outputs=None
    ) -> Dict[str, Any]:

        """Processes a single record using the guidance program.

        Args:
            record (dict or InternalDataFrame): The record to be processed.
            program (callable): The guidance program for processing.
            extra_fields (dict, optional): Additional fields to include in the processed record.
            outputs (list of str, optional): Specific output fields to extract from the result.

        Returns:
            dict: Processed output for the record.
        """
        
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
        if self.verbose:
            print_text(str(verified_input))
        result = program(
            silent=not self.verbose,
            **verified_input
        )
        if not outputs:
            verified_output = {'': str(result)}
        else:
            verified_output = {field: result[field] for field in outputs}

        return verified_output

    def get_input_program(self, input_template):
        """Generates an input program from the provided template.

        Args:
            input_template (str): Template to generate the input program.

        Returns:
            callable: The generated input program.
        """
        
        # fix input template in case "text" is presented there - there might be other paramater names as well...
        fixed_input_template = input_template
        if '{{text}}' in fixed_input_template:
            fixed_input_template = fixed_input_template.replace('{{text}}', '{{text_}}')
        input_program = guidance(fixed_input_template, llm=self._llm, silent=not self.verbose)
        return input_program

    def get_output_program(self, output_template):
        """Generates an output program from the provided template.

        Args:
            output_template (str): Template to generate the output program.

        Returns:
            callable: The generated output program.
        """
        
        output_program = guidance(output_template, llm=self._llm)
        return output_program

    def get_instructions_program(self, instructions):
        """Generates an instructions program from the provided template.

        Args:
            instructions (str): The instructions to generate the program.

        Returns:
            callable: The generated instructions program.
        """
        
        instructions_program = guidance(instructions, llm=self._llm)
        return instructions_program

    def _prepare_program_and_params(self, input_template, output_template, instructions, extra_fields):
        extra_fields = extra_fields or {}
        extra_fields = extra_fields.copy()
        # if only one program template is provided, use it as a program
        if output_template is None and instructions is None:
            program = self.get_input_program(input_template)
        else:
            program = self._program
            extra_fields.update({
                'input_program': self.get_input_program(input_template),
                'output_program': self.get_output_program(output_template),
                'instructions_program': self.get_instructions_program(instructions),
            })
        return program, extra_fields

    def process_record(
        self,
        record: Dict[str, Any],
        input_template: str,
        output_template: Optional[str] = None,
        instructions: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Processes a record using the provided templates and instructions.

        Args:
            record (Dict[str, Any]): The record data to be processed.
            input_template (str): Template for input processing.
            output_template (str): Template for output processing.
            instructions (str): Instructions for guidance.
            extra_fields (Dict[str, Any], optional): Additional fields to include during processing.

        Returns:
            Dict[str, Any]: The processed record.
        """
        outputs = self.get_outputs(output_template)
        program, extra_fields = self._prepare_program_and_params(input_template, output_template, instructions, extra_fields)
        output = self._process_record(
            record=record,
            program=program,
            outputs=outputs,
            extra_fields=extra_fields
        )
        return output

    def process_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        output_template: Optional[str] = None,
        instructions: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> InternalDataFrame:
        """Processes a batch of records using the provided templates and instructions.

        Args:
            batch (InternalDataFrame): The batch of records to be processed.
            input_template (str): Template for input processing.
            output_template (str): Template for output processing.
            instructions (str): Instructions for guidance.
            extra_fields (Dict[str, Any], optional): Additional fields to include during batch processing.

        Returns:
            InternalDataFrame: The processed batch of records.
        """
        
        outputs = self.get_outputs(output_template)
        program, extra_fields = self._prepare_program_and_params(input_template, output_template, instructions, extra_fields)
        output = batch.progress_apply(
            self._process_record,
            axis=1,
            result_type='expand',
            program=program,
            outputs=outputs,
            extra_fields=extra_fields
        )
        return output


class CodeRuntime(Runtime):
    """Base class representing a runtime designed for executing code."""
