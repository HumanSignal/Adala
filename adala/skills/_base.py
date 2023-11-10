from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict, Tuple, Union
from abc import ABC, abstractmethod
from adala.utils.internal_data import InternalDataFrame
from adala.utils.parse import parse_template, partial_str_format
from adala.runtimes.base import Runtime


Record = Dict[str, str]


class Skill(BaseModel, ABC):
    name: str = Field(
        title='Skill name',
        description='Unique name of the skill',
        examples=['labeling', 'classification', 'text-generation']
    )
    instructions: str = Field(
        title='Skill instructions',
        description='Instructs agent what to do with the input data. '
                    'Can use templating to refer to input fields.',
        examples=['Label the input text with the following labels: {labels}']
    )
    input_template: str = Field(
        title='Input template',
        description='Template for the input data. '
                    'Can use templating to refer to input parameters and perform data transformations.',
        examples=['Input: {input}', 'Input: {input}\nLabels: {labels}\nOutput: ']
    )
    output_template: str = Field(
        title='Output template',
        description='Template for the output data. '
                    'Can use templating to refer to input parameters and perform data transformations',
        examples=["Output: {output}", "{predictions}"]
    )
    description: Optional[str] = Field(
        default='',
        title='Skill description',
        description='Description of the skill. Can be used to retrieve skill from the library.',
        examples=['The skill to perform sentiment analysis on the input text.']
    )
    field_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        title='Field schema',
        description='JSON schema for the fields of the input and output data.',
        examples=[{
            "input": {"type": "string"},
            "output": {"type": "string"},
            "labels": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                }
            }
        }])

    def _get_extra_fields(self):
        """
        Retrieves fields that are not categorized as system fields.

        Returns:
            dict: A dictionary containing fields that are not system fields.
        """

        # TODO: more robust way to exclude system fields
        system_fields = {
            'name', 'description', 'input_template', 'output_template', 'instructions',
            'field_schema'}
        extra_fields = self.model_dump(exclude=system_fields)
        return extra_fields

    def get_output_fields(self):
        """
        Retrieves output fields.

        Returns:
            List[str]: A list of output fields.
        """
        extra_fields = self._get_extra_fields()
        # TODO: input fields are not considered - shall we disallow input fields in output template?
        output_fields = parse_template(partial_str_format(self.output_template, **extra_fields), include_texts=False)
        return [f['text'] for f in output_fields]


class TransformSkill(Skill):

    def apply(
        self,
        input: Union[InternalDataFrame, Record],
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a dataframe and returns another dataframe.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalDataFrame: The processed data.
        """

        if isinstance(input, dict):
            input = InternalDataFrame([input])
        return runtime.batch_to_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
        )


class SynthesisSkill(Skill):

    def apply(
        self,
        input: Record,
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a record and returns a dataframe.

        Args:
            input (Dict[str, str]): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.
        """
        return runtime.record_to_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
        )


class AnalysisSkill(Skill):

    def apply(
        self,
        input: Union[InternalDataFrame, Record],
        runtime: Runtime,
    ) -> Record:
        """
        Applies the skill to a dataframe and returns a record.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.
        """
        if isinstance(input, dict):
            input = InternalDataFrame([input])
        elif isinstance(input, InternalDataFrame):
            if len(input) > 1:
                raise ValueError('Input dataframe must contain only one record.')

        output = runtime.batch_to_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
        )
        return output.to_dict(orient='records')[0]
