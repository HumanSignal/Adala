from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict, Tuple, Union
from abc import ABC, abstractmethod
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from adala.utils.parse import parse_template, partial_str_format
from adala.utils.logs import print_dataframe, print_text
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
        instructions_first (bool): Flag indicating if instructions should be executed before input. Defaults to True.
        verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
    """

    name: str = Field(
        title="Skill name",
        description="Unique name of the skill",
        examples=["labeling", "classification", "text-generation"],
    )
    instructions: str = Field(
        title="Skill instructions",
        description="Instructs agent what to do with the input data. "
        "Can use templating to refer to input fields.",
        examples=["Label the input text with the following labels: {labels}"],
    )
    input_template: str = Field(
        title="Input template",
        description="Template for the input data. "
        "Can use templating to refer to input parameters and perform data transformations.",
        examples=["Input: {input}", "Input: {input}\nLabels: {labels}\nOutput: "],
    )
    output_template: str = Field(
        title="Output template",
        description="Template for the output data. "
        "Can use templating to refer to input parameters and perform data transformations",
        examples=["Output: {output}", "{predictions}"],
    )
    description: Optional[str] = Field(
        default="",
        title="Skill description",
        description="Description of the skill. Can be used to retrieve skill from the library.",
        examples=["The skill to perform sentiment analysis on the input text."],
    )
    field_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        title="Field schema",
        description="JSON schema for the fields of the input and output data.",
        examples=[
            {
                "input": {"type": "string"},
                "output": {"type": "string"},
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    },
                },
            }
        ],
    )
    instructions_first: bool = Field(
        default=True,
        title="Instructions first",
        description="Flag indicating if instructions should be shown before the input data.",
        examples=[True, False],
    )

    def _get_extra_fields(self):
        """
        Retrieves fields that are not categorized as system fields.

        Returns:
            dict: A dictionary containing fields that are not system fields.
        """

        # TODO: more robust way to exclude system fields
        system_fields = {
            "name",
            "description",
            "input_template",
            "output_template",
            "instructions",
            "field_schema",
        }
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
        output_fields = parse_template(
            partial_str_format(self.output_template, **extra_fields),
            include_texts=False,
        )
        return [f["text"] for f in output_fields]

    @abstractmethod
    def apply(self, input, runtime):
        """
        Base method for applying the skill.
        """

    @abstractmethod
    def improve(self, predictions, train_skill_output, feedback, runtime):
        """
        Base method for improving the skill.
        """


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
            instructions_first=self.instructions_first,
        )

    def improve(
        self,
        predictions: InternalDataFrame,
        train_skill_output: str,
        feedback,
        runtime: Runtime,
    ):
        """
        Improves the skill.

        Args:
            predictions (InternalDataFrame): The predictions made by the skill.
            train_skill_output (str): The name of the output field of the skill.
            feedback (InternalDataFrame): The feedback provided by the user.
            runtime (Runtime): The runtime instance to be used for processing (CURRENTLY SUPPORTS ONLY `OpenAIChatRuntime`).

        """

        fb = feedback.feedback.rename(
            columns=lambda x: x + "__fb" if x in predictions.columns else x
        )
        analyzed_df = fb.merge(predictions, left_index=True, right_index=True)

        examples = []

        for i, row in enumerate(analyzed_df.to_dict(orient="records")):
            # if fb marked as NaN, skip
            if not row[f"{train_skill_output}__fb"]:
                continue
            examples.append(
                f"### Example #{i}\n\n"
                f"{self.input_template.format(**row)}\n\n"
                f"{self.output_template.format(**row)}\n\n"
                f'User feedback: {row[f"{train_skill_output}__fb"]}\n\n'
            )

        examples = "\n".join(examples)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # full template
        if self.instructions_first:
            full_template = f"""
{{prompt}}
{self.input_template}
{self.output_template}"""
        else:
            full_template = f"""
{self.input_template}
{{prompt}}
{self.output_template}"""

        messages += [
            {
                "role": "user",
                "content": f"""
A prompt is a text paragraph that outlines the expected actions and instructs the large language model (LLM) to \
generate a specific output. This prompt is concatenated with the input text, and the \
model then creates the required output.
This describes the full template how the prompt is concatenated with the input to produce the output:

```
{full_template}
```

Here:
- "{self.input_template}" is input template,
- "{{prompt}}" is the LLM prompt,
- "{self.output_template}" is the output template.

Model can produce erroneous output if a prompt is not well defined. \
In our collaboration, we’ll work together to refine a prompt. The process consists of two main steps:

## Step 1
I will provide you with the current prompt along with prediction examples. Each example contains the input text, the final prediction produced by the model, and the user feedback. \
User feedback indicates whether the model prediction is correct or not. \
Your task is to analyze the examples and user feedback, determining whether the \
existing instruction is describing the task reflected by these examples precisely, and suggests changes to the prompt to address the incorrect predictions.

## Step 2
Next, you will carefully review your reasoning in step 1, integrate the insights to refine the prompt, \
and provide me with the new prompt that improves the model’s performance.""",
            }
        ]

        messages += [
            {
                "role": "assistant",
                "content": "Sure, I’d be happy to help you with this prompt engineering problem. "
                "Please provide me with the current prompt and the examples with user feedback.",
            }
        ]

        messages += [
            {
                "role": "user",
                "content": f"""
        ## Current prompt
        {self.instructions}

        ## Examples
        {examples}

        Summarize your analysis about incorrect predictions and suggest changes to the prompt.""",
            }
        ]

        reasoning = runtime.execute(messages)

        messages += [
            {"role": "assistant", "content": reasoning},
            {
                "role": "user",
                "content": f"""
Now please carefully review your reasoning in Step 1 and help with Step 2: refining the prompt.

## Current prompt
{self.instructions}

## Follow this guidance to refine the prompt:

1. The new prompt should should describe the task precisely, and address the points raised in the user feedback.

2. The new prompt should be similar to the current instruction, and only differ in the parts that address the issues you identified in Step 1.
    Example:
    - Current prompt: "The model should generate a summary of the input text."
    - New prompt: "The model should generate a summary of the input text. Pay attention to the original style."

3. Reply only with the new prompt. Do not include input and output templates in the prompt.""",
            },
        ]

        # display dialogue:
        for message in messages:
            print(f'"{{{message["role"]}}}":\n{message["content"]}')
        new_prompt = runtime.execute(messages)
        self.instructions = new_prompt


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
            instructions_first=self.instructions_first,
        )

    def improve(self, **kwargs):
        """
        Improves the skill.
        """
        raise NotImplementedError


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
        ).str.cat(sep="\n")

        output = runtime.record_to_record(
            {"input": aggregated_input},
            input_template="{input}",
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
            instructions_first=self.instructions_first,
        )
        # output['input'] = aggregated_input
        # concatenate input and output and return dataframe
        return InternalSeries(output)

    def improve(self, **kwargs):
        """
        Improves the skill.
        """
        raise NotImplementedError
