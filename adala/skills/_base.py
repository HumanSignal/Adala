import logging
import string
import traceback
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    field_serializer,
)
from typing import List, Optional, Any, Dict, Tuple, Union, ClassVar, Type
from abc import ABC, abstractmethod
from adala.utils.internal_data import (
    InternalDataFrame,
    InternalDataFrameConcat,
    InternalSeries,
)
from adala.utils.parse import parse_template, partial_str_format
from adala.utils.pydantic_generator import field_schema_to_pydantic_class
from adala.utils.logs import print_dataframe, print_text
from adala.utils.registry import BaseModelInRegistry
from adala.runtimes.base import Runtime, AsyncRuntime
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Skill(BaseModelInRegistry):
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
        frozen (bool): Flag indicating if the skill is frozen. Defaults to False.
        response_model (Optional[Type[BaseModel]]): Pydantic-based response model for the skill. If used, `output_template` and `field_schema` are ignored. Note that using `response_model` will become the default in the future.
        type (ClassVar[str]): Type of the skill.
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
        # TODO: instructions can be deprecated in favor of using `input_template` to specify the instructions
        default="",
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
        # TODO: output_template can be deprecated in favor of using `response_model` to specify the output
        default="",
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

    frozen: bool = Field(
        default=False,
        title="Frozen",
        description="Flag indicating if the skill is frozen.",
        examples=[True, False],
    )

    response_model: Type[BaseModel] = Field(
        default=None,
        title="Response model",
        description="Pydantic-based response model for the skill. If used, `output_template` and `field_schema` are ignored.",
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
            "extra_fields",
            "instructions_first",
            "verbose",
            "frozen",
            "response_model",
            "type",
        }
        extra_fields = self.model_dump(exclude=system_fields)
        return extra_fields

    def get_input_fields(self):
        """
        Retrieves input fields.

        Returns:
            List[str]: A list of input fields.
        """
        extra_fields = self._get_extra_fields()
        input_fields = parse_template(
            partial_str_format(self.input_template, **extra_fields),
            include_texts=False,
        )
        return [f["text"] for f in input_fields]

    def get_output_fields(self):
        """
        Retrieves output fields.

        Returns:
            List[str]: A list of output fields.
        """
        if self.response_model:
            return list(self.response_model.__fields__.keys())
        if self.field_schema:
            return list(self.field_schema.keys())

        extra_fields = self._get_extra_fields()
        # TODO: input fields are not considered - shall we disallow input fields in output template?
        output_fields = parse_template(
            partial_str_format(self.output_template, **extra_fields),
            include_texts=False,
        )
        return [f["text"] for f in output_fields]

    def _create_response_model_from_field_schema(self):
        assert self.field_schema, "field_schema is required to create a response model"
        if self.response_model:
            return
        self.response_model = field_schema_to_pydantic_class(
            self.field_schema, self.name, self.description
        )

    @model_validator(mode="after")
    def validate_response_model(self):
        if self.response_model:
            # if response_model, we use it right away
            return self

        if not self.field_schema:
            # if field_schema is not provided, extract it from `output_template`
            logger.info(
                f"Parsing output_template to generate the response model: {self.output_template}"
            )
            self.field_schema = {}
            chunks = parse_template(self.output_template)

            previous_text = ""
            for chunk in chunks:
                if chunk["type"] == "text":
                    previous_text = chunk["text"]
                if chunk["type"] == "var":
                    field_name = chunk["text"]
                    # by default, all fields are strings
                    field_type = "string"

                    # if description is not provided, use the text before the field,
                    # otherwise use the field name with underscores replaced by spaces
                    field_description = previous_text or field_name.replace("_", " ")
                    field_description = field_description.strip(
                        string.punctuation
                    ).strip()
                    previous_text = ""

                    # create default JSON schema entry for the field
                    self.field_schema[field_name] = {
                        "type": field_type,
                        "description": field_description,
                    }

        self._create_response_model_from_field_schema()
        return self

    # When serializing the agent, ensure `response_model` is excluded.
    # It will be restored from `field_schema` during deserialization.
    @field_serializer("response_model")
    def serialize_response_model(self, value):
        return None

    # remove `response_model` from the pickle serialization
    def __getstate__(self):
        state = super().__getstate__()
        state["__dict__"]["response_model"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # ensure response_model is restored from field_schema, if not already set
        self._create_response_model_from_field_schema()

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
            response_model=self.response_model,
        )

    async def aapply(
        self,
        input: InternalDataFrame,
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a dataframe and returns another dataframe.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalDataFrame: The transformed data.
        """

        return await runtime.batch_to_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions_template=self.instructions,
            field_schema=self.field_schema,
            extra_fields=self._get_extra_fields(),
            instructions_first=self.instructions_first,
            response_model=self.response_model,
        )

    def improve(
        self,
        predictions: InternalDataFrame,
        train_skill_output: str,
        feedback,
        runtime: Runtime,
        add_cot: bool = False,
    ):
        """
        Improves the skill.

        Args:
            predictions (InternalDataFrame): The predictions made by the skill.
            train_skill_output (str): The name of the output field of the skill.
            feedback (InternalDataFrame): The feedback provided by the user.
            runtime (Runtime): The runtime instance to be used for processing (CURRENTLY SUPPORTS ONLY `OpenAIChatRuntime`).
            add_cot (bool): Flag indicating if the skill should be used the Chain-of-Thought strategy. Defaults to False.
        """
        if feedback.feedback[train_skill_output].isna().all():
            # No feedback left - nothing to improve
            return

        if feedback.match[train_skill_output].all():
            # all feedback is "correct" - nothing to improve
            return

        fb = feedback.feedback.rename(
            columns=lambda x: x + "__fb" if x in predictions.columns else x
        )
        analyzed_df = fb.merge(predictions, left_index=True, right_index=True)

        examples = []

        for i, row in enumerate(analyzed_df.to_dict(orient="records")):
            # if fb marked as NaN, skip
            if not row[f"{train_skill_output}__fb"]:
                continue

            # TODO: self.output_template can be missed or incompatible with the field_schema
            # we need to redefine how we create examples for learn()
            if not self.output_template:
                raise ValueError(
                    "`output_template` is required for improve() method and must contain "
                    "the output fields from `field_schema`"
                )
            examples.append(
                f"### Example #{i}\n\n"
                f"{partial_str_format(self.input_template, **row)}\n\n"
                f"{partial_str_format(self.output_template, **row)}\n\n"
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
        reasoning = runtime.get_llm_response(messages)

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

2. The new prompt should be similar to the current prompt, and only differ in the parts that address the issues you identified in Step 1.
    Example:
    - Current prompt: "Generate a summary of the input text."
    - New prompt: "Generate a summary of the input text. Pay attention to the original style."

3. Reply only with the new prompt. Do not include input and output templates in the prompt.
""",
            },
        ]

        if add_cot:
            cot_instructions = """

4. In the new prompt, you should ask the model to perform step-by-step reasoning, and provide rationale or explanations for its prediction before giving the final answer. \
Instruct the model to give the final answer at the end of the prompt, using the following template: "Final answer: <answer>".
    Example:
    - Current prompt: "Generate a summary of the input text."
    - New prompt: "Generate a summary of the input text. Explain your reasoning step-by-step. Use the following template to give the final answer at the end of the prompt: "Final answer: <answer>"."""
            messages[-1]["content"] += cot_instructions
        # display dialogue:
        for message in messages:
            print(f'"{{{message["role"]}}}":\n{message["content"]}')
        new_prompt = runtime.get_llm_response(messages)
        self.instructions = new_prompt

    async def aimprove(
        self,
        teacher_runtime: AsyncRuntime,
        target_input_variables: List[str],
        predictions: Optional[InternalDataFrame] = None,
        instructions: Optional[str] = None,
    ):
        """
        Improves the skill.
        """

        from adala.skills.collection.prompt_improvement import (
            PromptImprovementSkill,
            ImprovedPromptResponse,
            ErrorResponseModel,
            PromptImprovementSkillResponseModel,
        )

        response_dct = {}
        try:
            prompt_improvement_skill = PromptImprovementSkill(
                skill_to_improve=self,
                input_variables=target_input_variables,
                instructions=instructions,
            )
            if predictions is None:
                input_df = InternalDataFrame()
            else:
                input_df = predictions
            response_df = await prompt_improvement_skill.aapply(
                input=input_df,
                runtime=teacher_runtime,
            )

            # awkward to go from response model -> dict -> df -> dict -> response model
            response_dct = response_df.iloc[0].to_dict()

            # unflatten the response
            if response_dct.pop("_adala_error", False):
                output = ErrorResponseModel(**response_dct)
            else:
                output = PromptImprovementSkillResponseModel(**response_dct)

        except Exception as e:
            logger.error(
                f"Error improving skill: {e}. Traceback: {traceback.format_exc()}"
            )
            output = ErrorResponseModel(
                _adala_message=str(e),
                _adala_details=traceback.format_exc(),
            )

        # get tokens and token cost
        resp = ImprovedPromptResponse(output=output, **response_dct)
        logger.debug(f"resp: {resp}")

        return resp


class SampleTransformSkill(TransformSkill):
    sample_size: int

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a dataframe and returns a dataframe.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalDataFrame: The processed data.
        """
        return super(SampleTransformSkill, self).apply(
            input.sample(self.sample_size), runtime
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
            instructions_first=self.instructions_first,
            response_model=self.response_model,
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

    input_prefix: str = ""
    input_separator: str = "\n"
    chunk_size: Optional[int] = None

    def _iter_over_chunks(
        self, input: InternalDataFrame, chunk_size: Optional[int] = None
    ):
        """
        Iterates over chunks of the input dataframe.
        Returns a generator of strings that are the concatenation of the rows of the chunk with `input_separator`
        interpolated with the `input_template` and `extra_fields`.
        """

        if input.empty:
            yield ""
            return

        if isinstance(input, InternalSeries):
            input = input.to_frame()
        elif isinstance(input, dict):
            input = InternalDataFrame([input])

        extra_fields = self._get_extra_fields()
        
        # if chunk_size is specified, split the input into chunks and process each chunk separately
        if self.chunk_size is not None:
            chunks = (
                input.iloc[i : i + self.chunk_size]
                for i in range(0, len(input), self.chunk_size)
            )
        else:
            chunks = [input]
            
        # define the row preprocessing function
        def row_preprocessing(row):
            return partial_str_format(self.input_template, **row, **extra_fields, i=int(row.name) + 1)

        total = input.shape[0] // self.chunk_size if self.chunk_size is not None else 1
        for chunk in tqdm(chunks, desc="Processing chunks", total=total):
            # interpolate every row with input_template and concatenate them with input_separator to produce a single string
            agg_chunk = (
                chunk.reset_index()
                .apply(row_preprocessing, axis=1)
                .str.cat(sep=self.input_separator)
            )
            yield agg_chunk

    def apply(
        self,
        input: Union[InternalDataFrame, InternalSeries, Dict],
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a dataframe and returns a record.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.

        Returns:
            InternalSeries: The record containing the analysis results.
        """
        outputs = []
        for agg_chunk in self._iter_over_chunks(input):
            output = runtime.record_to_record(
                {"input": f"{self.input_prefix}{agg_chunk}"},
                input_template="{input}",
                output_template=self.output_template,
                instructions_template=self.instructions,
                instructions_first=self.instructions_first,
                response_model=self.response_model,
            )
            outputs.append(InternalSeries(output))
        output = InternalDataFrame(outputs)

        return output

    async def aapply(
        self,
        input: Union[InternalDataFrame, InternalSeries, Dict],
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:
        """
        Applies the skill to a dataframe and returns a record.
        """
        outputs = []
        for agg_chunk in self._iter_over_chunks(input):
            output = await runtime.record_to_record(
                {"input": f"{self.input_prefix}{agg_chunk}"},
                input_template="{input}",
                output_template=self.output_template,
                instructions_template=self.instructions,
                instructions_first=self.instructions_first,
                response_model=self.response_model,
            )
            outputs.append(InternalSeries(output))
        output = InternalDataFrame(outputs)

        return output

    def improve(self, **kwargs):
        """
        Improves the skill.
        """
        raise NotImplementedError
