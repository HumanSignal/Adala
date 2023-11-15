from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict, Tuple, Union
from abc import ABC, abstractmethod
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from adala.utils.parse import parse_template, partial_str_format
from adala.runtimes.base import Runtime


class Skill(BaseModel, ABC):
    """
    Abstract base class representing a skill.

    Provides methods to interact with and obtain information about skills.

    Attributes:
        name (str): Unique name of the skill.
        instructions (str): Instructs agent what to do with the input data.
        input_template (str): Template for the input data.
        output_template (str): Template for the output data.
        description (Optional[str]): Description of the skill.
        field_schema (Optional[Dict]): Field [JSON schema](https://json-schema.org/) to use in the templates. Defaults to all fields are strings,
            i.e. analogous to {"field_n": {"type": "string"}}.
        extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
        verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
    """

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
    """
    Transform skill that transforms a dataframe to another dataframe (e.g. for data annotation purposes).
    See base class Skill for more information about the attributes.
    """
    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a dataframe and returns another dataframe.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalDataFrame: The transformed data.
        """

        return runtime.batch_to_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
        )


class SynthesisSkill(Skill):
    """
    Synthesis skill that synthesize a dataframe from a record (e.g. for dataset generation purposes).
    See base class Skill for more information about the attributes.
    """

    def apply(
        self,
        input: Union[Dict, InternalSeries],
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a record and returns a dataframe.

        Args:
            input (InternalSeries): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalDataFrame: The synthesized data.
        """
        if isinstance(input, InternalSeries):
            input = input.to_dict()
        return runtime.record_to_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
        )


class AnalysisSkill(Skill):
    """
    Analysis skill that analyzes a dataframe and returns a record (e.g. for data analysis purposes).
    See base class Skill for more information about the attributes.
    """

    def apply(
        self,
        input: Union[InternalDataFrame, InternalSeries, Dict],
        runtime: Runtime,
    ) -> InternalSeries:
        """
        Applies the skill to a dataframe and returns a record.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalSeries: The record containing the analysis results.
        """
        if isinstance(input, InternalSeries):
            input = input.to_frame()
        elif isinstance(input, dict):
            input = InternalDataFrame([input])

        extra_fields = self._get_extra_fields()

        aggregated_input = input.apply(
            lambda row: self.input_template.format(**row, **extra_fields), axis=1
        ).str.cat(sep='\n')

        output = runtime.record_to_record(
            {'aggregated_input': aggregated_input},
            input_template='{aggregated_input}',
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
        )
        return InternalSeries(output)
