import openai
import pandas as pd
import re

from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Tuple
from abc import ABC, abstractmethod
from pydantic import Field, field_serializer, model_validator

from typing import Optional
from adala.runtimes.base import LLMRuntime
from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import ShortTermMemory, LongTermMemory
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat


class BaseSkill(BaseModel, ABC):
    """
    """
    name: str = Field(
        title='Skill name',
        description='Unique name of the skill',
        default='',
        examples=['labeling', 'classification', 'text-generation']
    )
    instructions: str = Field(
        title='Skill instructions',
        description='Instructs agent what to do with the input data. '
                    'Can use templating to refer to input fields.',
        default='',
        examples=['Label the input text with the following labels: {{{{labels}}}}']
    )
    description: Optional[str] = Field(
        default='',
        title='Skill description',
        description='Description of the skill. Can be used to retrieve skill from the library.',
        examples=['The skill to perform sentiment analysis on the input text.']
    )
    input_template: Optional[str] = Field(
        title='Input template',
        description='Template for the input data. '
                    'Can use templating to refer to input parameters and perform data transformations.',
        default="Input: {{{{{input}}}}}",
        examples=["Text: {{{{text_column}}}}, Date: {{{{date_column}}}}, Sentiment: {{{{gen 'sentiment'}}}}"]
    )
    output_template: Optional[str] = Field(
        title='Output template',
        description='Template for the output data. '
                    'Can use templating to refer to input parameters and perform data transformations. '
                    'Should contain at least one field matching `validation_fields`.',
        default="Output: {{{{gen 'predictions'}}}}",
        examples=["Output: {{{{select 'predictions' options=labels logprobs='score'}}}}"]
    )
    validation_fields: Optional[List[str]] = Field(
        title='Prediction fields',
        description='List of fields that will require validation. '
                    'Should match at least one field in `output_template`.',
        examples=['predictions', 'more_predictions'],
        default=['predictions']
    )

    def __call__(self, input: InternalDataFrame, runtime: Runtime, dataset: Dataset) -> InternalDataFrame:
        """
        Call runtime to process batch of inputs.
        Input and output shapes can be varying.
        This method is supposed to be the main way of connecting different skills together.
        It should also take care of input / output data types validation.
        """

        # get user defined dataset input fields
        input_template, output_template, instructions = self._get_formatted_templates(dataset)

        runtime_predictions = runtime.process_batch(
            batch=input,
            input_template=input_template,
            output_template=output_template,
            instructions=instructions,
            extra_fields=self._get_extra_fields()
        )
        return InternalDataFrameConcat((input, runtime_predictions), axis=1)

    def _get_formatted_templates(self, dataset: Dataset) -> Tuple[str, str, str]:
        """
        Format input and output templates with dataset input fields
        """
        inputs = self._get_extra_fields()
        if dataset.input_data_field:
            inputs['input'] = dataset.input_data_field
        input_template = self.input_template.format(**inputs)
        output_template = self.output_template.format(**inputs)
        instructions = self.instructions.format(**inputs)
        return input_template, output_template, instructions

    def _get_extra_fields(self):
        # TODO: more robust way to exclude system fields
        system_fields = {
            'name', 'description', 'input_template', 'output_template', 'instructions', 'validation_fields'}
        extra_fields = self.model_dump(exclude=system_fields)
        return extra_fields

    @abstractmethod
    def apply(
        self, dataset: Dataset,
        runtime: Runtime,
        experience: ShortTermMemory
    ) -> ShortTermMemory:
        """
        Apply skill to dataset and return new dataset with skill results (predictions)
        """

    @abstractmethod
    def analyze(
        self, experience: ShortTermMemory,
        memory: Optional[LongTermMemory] = None,
        runtime: Optional[Runtime] = None
    ) -> ShortTermMemory:
        """
        Analyze results and return observed experience
        Agent can optionally retrieve long term memory to enrich experience
        """

    @abstractmethod
    def improve(self, experience: ShortTermMemory) -> ShortTermMemory:
        """
        Improve current skill state based on current experience
        """


class LLMSkill(BaseSkill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """

    def apply(
        self, dataset: Dataset,
        runtime: LLMRuntime,
        experience: ShortTermMemory
    ) -> ShortTermMemory:

        experience = experience.model_copy()

        predictions = []

        for batch in dataset.batch_iterator():
            runtime_predictions = self(batch, runtime, dataset)
            predictions.append(runtime_predictions)

        if not predictions:
            predictions = InternalDataFrame()
        else:
            predictions = InternalDataFrameConcat(predictions, copy=False)

        # append predictions to existing experience, to chain skills
        # TODO: implement predictions chaining
        experience.predictions = predictions
        # if experience.predictions is None:
        #     experience.predictions = predictions
        # else:
        #     experience.predictions = InternalDataFrameConcat([
        #         experience.predictions.drop(columns=[col for col in experience.predictions.columns if col in predictions.columns]),
        #         predictions
        #     ], axis=1)
        #     raise NotImplementedError

        return experience

    def analyze(
        self, experience: ShortTermMemory,
        memory: Optional[LongTermMemory] = None,
        runtime: Optional[Runtime] = None
    ) -> ShortTermMemory:

        experience = experience.model_copy()

        # TODO: can be multiple prediction validation fields
        match = experience.match_column_name
        errors = experience.evaluations[~experience.evaluations[match]]
        experience.accuracy = experience.evaluations[match].mean()
        if errors.empty:
            # No errors - nothing to analyze
            experience.errors = errors
            return experience

        # collect errors and create error report
        # first sample errors - make it uniform, but more sophisticated sampling can be implemented
        errors = errors.sample(n=min(3, errors.shape[0]))

        # collect error inputs from runtime
        input_template, _, instructions = self._get_formatted_templates(experience.dataset)
        extra_fields = self._get_extra_fields()
        inputs = runtime.process_batch_inputs(
            errors,
            input_template,
            extra_fields=extra_fields
        )

        # construct error report
        errors = pd.concat([
            inputs,
            errors[[experience.prediction_column_name, experience.ground_truth_column_name]]
        ], axis=1)
        errors.columns = ['input', 'prediction', 'ground_truth']
        smart_runtime = LLMRuntime(llm_params={'model': 'gpt-4'}, verbose=True)
        error_reasons = smart_runtime.process_batch(
            errors,
            instructions="{{#system~}}\n"
                         "LLM prompt was created by concatenating instructions with text input:\n\n"
                         "Prediction = LLM(Input, Instructions)\n\n"
                         "We expect the prediction to be equal to the ground truth.\n"
                         "Your task is to provide a reason for the error due to the original instruction.\n"
                         "Be concise and specific.\n\n"
                         f"Instructions: {instructions}\n"
                         "{{~/system}}",
            input_template="{{#user~}}\n"
                           "Input: {{input}}\n"
                           "Prediction: {{prediction}}\n"
                           "Ground truth: {{ground_truth}}\n"
                           "Explanation:\n"
                           "{{~/user}}",
            output_template="{{#assistant~}}{{gen 'reason'}}{{~/assistant}}",
            extra_fields=extra_fields
        )
        errors['reason'] = error_reasons['reason']

        experience.errors = errors
        return experience

    def improve(self, experience: ShortTermMemory) -> ShortTermMemory:
        experience = experience.model_copy()

        errors = experience.errors.to_dict(orient='records')
        smart_runtime = LLMRuntime(llm_params={'model': 'gpt-4'}, verbose=True)
        _, _, instructions = self._get_formatted_templates(experience.dataset)
        result = smart_runtime.process_record(
            record={
                'errors': errors
            },
            instructions="{{#system~}}\n"
                         "LLM prompt was created by concatenating instructions with text input:\n\n"
                         "Prediction = LLM(Input, Instructions)\n\n"
                         "We expect the prediction to be equal to the ground truth.\n"
                         "Your task is to craft a revised concise instruction for the LLM. "
                         "Follow best practices for LLM prompt engineering.\n"
                         "Include 2-3 examples at the end of your response to demonstrate how the new instruction would be applied.\n"
                         "Use the following format for your examples:\n"
                         "Input: ...\n"
                         "Output: ...\n\n"
                         "{{~/system}}\n",
            input_template="{{#user~}}\n"
                           f"Old instruction: {instructions}\n"
                           "Errors: {{#each errors}}"
                           "\nInput: {{this.input}}\n"
                           "Prediction: {{this.prediction}}\n"
                           "Ground truth: {{this.ground_truth}}\n"
                           "{{/each}}\n"
                           "New instruction:\n"
                           "{{~/user}}",
            output_template="{{#assistant~}}{{gen 'new_instruction'}}{{~/assistant}}",
            extra_fields=self._get_extra_fields()
        )
        new_instruction = result['new_instruction']

        experience.initial_instructions = self.instructions
        experience.updated_instructions = new_instruction
        return experience
