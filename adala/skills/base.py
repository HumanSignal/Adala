import openai
import pandas as pd
import re

from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Tuple
from abc import ABC, abstractmethod
from pydantic import Field, model_validator

from typing import Optional
from adala.runtimes.base import LLMRuntime
from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import ShortTermMemory, LongTermMemory
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat


class BaseSkill(BaseModel, ABC):
    """
    A foundational abstract class representing a skill. This class sets the foundation 
    for all skills and provides common attributes and methods for skill-based operations.
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
        examples=['Label the input text with the following labels: {{labels}}']
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
        examples=["Text: {{{{{input}}}}}, Date: {{{{date_column}}}}, Sentiment: {{{{gen 'sentiment'}}}}"]
    )
    input_data_field: Optional[str] = Field(
        title='Input data field',
        description='Input data field name that will be used to match input data.',
        examples=['text'],
        # TODO: either make it required, or `input_template` required
        default='text'
    )
    output_template: Optional[str] = Field(
        title='Output template',
        description='Template for the output data. '
                    'Can use templating to refer to input parameters and perform data transformations. '
                    'Should contain at least one field matching `validation_fields`.',
        default="Output: {{gen 'predictions'}}",
        examples=["Output: {{select 'predictions' options=labels logprobs='score'}}"]
    )
    prediction_field: Optional[str] = Field(
        title='Prediction field',
        description='Prediction field name that will be used to match ground truth labels.'
                    'Should match at least one output field in `output_template`, e.g. \'predictions\'',
        examples=['predictions'],
        default='predictions'
    )

    @model_validator(mode='after')
    def validate_input_template(self):
        """
        Validates the input_template, updating it if necessary.
        
        Returns:
            BaseSkill: Updated instance of the BaseSkill class.
        """
        
        if '{{{{{input}}}}}' in self.input_template:
            # TODO: check why it is called multiple times
            self.input_template = self.input_template.format(input=self.input_data_field)
        return self

    def __call__(self, input: InternalDataFrame, runtime: Runtime, dataset: Dataset) -> InternalDataFrame:
        """Calls the runtime to process a batch of inputs. Input and
        output shapes can be varying, and it should also take care of
        data types validation

        Args:
            input (InternalDataFrame): Input data in the form of an InternalDataFrame.
            runtime (Runtime): The runtime instance to be used for processing.
            dataset (Dataset): The dataset containing the data to be processed.
        
        Returns:
            InternalDataFrame: Concatenated dataframe with the original input and the predictions from the runtime.

        """

        # get user defined dataset input fields

        runtime_predictions = runtime.process_batch(
            batch=input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions=self.instructions,
            extra_fields=self._get_extra_fields()
        )
        return InternalDataFrameConcat((input, runtime_predictions), axis=1)

    def _get_extra_fields(self):
        """
        Retrieves fields that are not categorized as system fields.
        
        Returns:
            dict: A dictionary containing fields that are not system fields.
        """
        
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
        Applies the skill to a dataset and returns the results.
        
        Args:
            dataset (Dataset): The dataset on which the skill is to be applied.
            runtime (Runtime): The runtime instance to be used for processing.
            experience (ShortTermMemory): Previous experiences or results.
        
        Returns:
            ShortTermMemory: The updated experience after applying the skill.
        """        

    @abstractmethod
    def analyze(
        self, experience: ShortTermMemory,
        memory: Optional[LongTermMemory] = None,
        runtime: Optional[Runtime] = None
    ) -> ShortTermMemory:
        """
        Analyzes the results to derive new experiences.
        
        Args:
            experience (ShortTermMemory): The current experience.
            memory (LongTermMemory, optional): Previous long term memories. Defaults to None.
            runtime (Runtime, optional): The runtime instance. Defaults to None.
        
        Returns:
            ShortTermMemory: The updated experience after analysis.
        """

    @abstractmethod
    def improve(self, experience: ShortTermMemory, update_instructions: bool = True) -> ShortTermMemory:
        """
        Refines the current state of the skill based on its experiences.
        
        Args:
            experience (ShortTermMemory): The current experience.
            update_instructions (bool, optional): Flag to decide if instructions should be updated. Defaults to True.
        
        Returns:
            ShortTermMemory: The updated experience after improvements.
        """


class LLMSkill(BaseSkill):
    """
    A skill specialized for Language Models (LLM). Inherits from the BaseSkill 
    class and provides specific implementations for handling LLM predictions based 
    on given instructions.
    """

    def apply(
        self,
        dataset: Dataset,
        runtime: LLMRuntime,
        experience: ShortTermMemory
    ) -> ShortTermMemory:
        """
        Applies the LLM skill on a dataset and returns the results.
        
        Args:
            dataset (Dataset): The dataset on which the skill is to be applied.
            runtime (LLMRuntime): The runtime instance to be used for processing.
            experience (ShortTermMemory): Previous experiences or results.
        
        Returns:
            ShortTermMemory: The updated experience after applying the skill.
        """
        
        experience = experience.model_copy()

        predictions = []

        for batch in dataset.batch_iterator():
            runtime_predictions = self(batch, runtime, dataset)
            predictions.append(runtime_predictions)

        if not predictions:
            predictions = InternalDataFrame()
        else:
            predictions = InternalDataFrameConcat(predictions, copy=False)
            predictions.rename(columns={self.prediction_field: self.name}, inplace=True)

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
        """
        Analyzes the results to identify any discrepancies and returns the observed experience.
        
        Args:
            experience (ShortTermMemory): The current experience.
            memory (LongTermMemory, optional): Previous long term memories. Defaults to None.
            runtime (Runtime, optional): The runtime instance. Defaults to None.
        
        Returns:
            ShortTermMemory: The updated experience after analysis.
        """
        
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
        extra_fields = self._get_extra_fields()
        inputs = runtime.process_batch_inputs(
            batch=errors,
            input_template=self.input_template,
            extra_fields=extra_fields
        )

        # construct error report
        errors = pd.concat([
            inputs,
            errors[[self.name, experience.ground_truth_column_name]]
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
                         f"Instructions: {self.instructions}\n"
                         "{{~/system}}",
            input_template="{{#user~}}\n"
                           "{{input}}\n"
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

    def improve(self, experience: ShortTermMemory, update_instructions: bool = True) -> ShortTermMemory:
        """
        Refines the LLM skill based on its recent experiences.
        
        Args:
            experience (ShortTermMemory): The current experience.
            update_instructions (bool, optional): Flag to decide if instructions should be updated. Defaults to True.
        
        Returns:
            ShortTermMemory: The updated experience after improvements.
        """
        
        experience = experience.model_copy()

        errors = experience.errors.to_dict(orient='records')
        smart_runtime = LLMRuntime(llm_params={'model': 'gpt-4'}, verbose=True)
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
                           f"Old instruction: {self.instructions}\n\n"
                           "Errors:\n{{#each errors}}"
                           "\n{{this.input}}\n"
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

        if update_instructions:
            self.instructions = new_instruction

        return experience
