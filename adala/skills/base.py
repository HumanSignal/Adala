import openai
import pandas as pd
import re

from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod
from pydantic import Field, field_serializer, model_validator

from typing import Optional
from adala.datasets.base import InternalDataFrame, InternalDataFrame_encoder
from adala.runtimes.base import LLMRuntime

from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import ShortTermMemory, LongTermMemory


class BaseSkill(BaseModel, ABC):
    name: str = Field(default='')
    description: Optional[str] = Field(default='')
    instructions: str = Field(default='')
    input_template: str = Field(default='')
    output_template: str = Field(default='')
    # TODO: how to work with multiple outputs?
    prediction_field: str = Field(default='')


    def __call__(self, input: InternalDataFrame, runtime: Runtime) -> InternalDataFrame:
        """
        Call runtime to process batch of inputs.
        Input and output shapes can be varying.
        This method is supposed to be the main way of connecting different skills together.
        It should also take care of input / output data types validation.
        """
        return self._call(input, runtime)

    @abstractmethod
    def _call(self, input: InternalDataFrame, runtime: Runtime) -> InternalDataFrame:
        """
        Apply skill to input data and return output data
        """

    @abstractmethod
    def apply(self, dataset: Dataset, runtime: Runtime) -> ShortTermMemory:
        """
        Apply skill to dataset and return new dataset with skill results (predictions)
        """

    @abstractmethod
    def evaluate(self, experience: ShortTermMemory) -> ShortTermMemory:
        """
        Test predictions and return new Dataset with evaluation results - e.g. error examples
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

    def learn(self, dataset: Dataset, runtime: Runtime, memory: Optional[LongTermMemory] = None) -> ShortTermMemory:
        """
        Apply, validate, analyze and optimize skill.
        """
        experience = self.apply(dataset=dataset, runtime=runtime)
        print('Evaluating, analyzing and improving...')
        experience = self.evaluate(experience)
        experience = self.analyze(experience, memory, runtime)
        experience = self.improve(experience)
        print('Done!')
        return experience


class Skill(BaseSkill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """

    def _get_extra_fields(self) -> Dict[str, Any]:
        extra_fields = self.model_dump(
            # TODO: more robust way to exclude system fields
            exclude={'name', 'description', 'input_template', 'output_template', 'instructions', 'prediction_field'})
        return extra_fields

    def _call(self, input: InternalDataFrame, runtime: Runtime) -> InternalDataFrame:

        runtime_outputs = runtime.process_batch(
            input,
            input_template=self.input_template,
            output_template=self.output_template,
            instructions=self.instructions,
            extra_fields=self._get_extra_fields()
        )
        return runtime_outputs

    def apply(self, dataset: Dataset, runtime: LLMRuntime) -> ShortTermMemory:
        predictions = []

        for batch in dataset.batch_iterator(
            # this is the current OpenAI limit
            batch_size=20
        ):
            runtime_outputs = self._call(batch, runtime)
            predictions.append(runtime_outputs)

        experience = ShortTermMemory(dataset=dataset)

        if not predictions:
            experience.predictions = pd.DataFrame()
        else:
            experience.predictions = pd.concat(predictions, copy=False)

        return experience

    def evaluate(self, experience) -> ShortTermMemory:
        gt = experience.dataset.get_ground_truth(experience.predictions)
        pred = experience.predictions.loc[gt.index]
        pred = pred[pred.notna()]

        # TODO: implement more sophisticated evaluation beyond simple equality
        match = pred[self.prediction_field] == gt[experience.dataset.ground_truth_column]
        pred[f'{self.prediction_field}_match'] = match
        evaluations = pd.concat([gt, pred], axis=1)
        updated_experience = experience.model_copy()
        updated_experience.evaluations = evaluations
        return updated_experience

    def analyze(
        self, experience: ShortTermMemory,
        memory: Optional[LongTermMemory] = None,
        runtime: Optional[Runtime] = None
    ) -> ShortTermMemory:
        errors = experience.evaluations[~experience.evaluations[f'{self.prediction_field}_match']]
        accuracy = experience.evaluations[f'{self.prediction_field}_match'].mean()

        # collect errors and create error report
        # first sample errors - make it uniform, but more sophisticated sampling can be implemented
        errors = errors.sample(n=min(3, errors.shape[0]))
        # collect error inputs from runtime
        inputs = runtime.process_batch_inputs(
            errors,
            self.input_template,
            extra_fields=self._get_extra_fields()
        )
        errors = pd.concat((inputs, errors[[self.prediction_field, experience.dataset.ground_truth_column]]), axis=1)
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
                         "Instructions: {{llm_instructions}}\n"
                         "{{~/system}}",
            input_template="{{#user~}}\n"
                           "Input: {{input}}\n"
                           "Prediction: {{prediction}}\n"
                           "Ground truth: {{ground_truth}}\n"
                           "Explanation:\n"
                           "{{~/user}}",
            output_template="{{#assistant~}}{{gen 'reason'}}{{~/assistant}}",
            extra_fields={'llm_instructions': self.instructions}
        )
        errors['reason'] = error_reasons['reason']

        updated_experience = experience.model_copy()
        updated_experience.errors = errors
        updated_experience.accuracy = accuracy
        return updated_experience

    def improve(self, experience: ShortTermMemory) -> ShortTermMemory:
        errors = experience.errors.to_dict(orient='records')
        smart_runtime = LLMRuntime(llm_params={'model': 'gpt-4'}, verbose=True)
        result = smart_runtime.process_record(
            record={
                'old_instruction': self.instructions,
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
                           "Old instruction: {{old_instruction}}\n"
                           "Errors: {{#each errors}}"
                           "\nInput: {{this.input}}\n"
                           "Prediction: {{this.prediction}}\n"
                           "Ground truth: {{this.ground_truth}}\n"
                           "{{/each}}\n"
                           "New instruction:\n"
                           "{{~/user}}",
            output_template="{{#assistant~}}{{gen 'new_instruction'}}{{~/assistant}}",
        )
        new_instruction = result['new_instruction']

        updated_experience = experience.model_copy()
        updated_experience.initial_instructions = self.instructions
        updated_experience.updated_instructions = new_instruction
        return updated_experience
