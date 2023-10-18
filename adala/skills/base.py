import openai
import pandas as pd
import re

from pydantic import BaseModel
from typing import List, Optional, Any
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
    prompt_template: str = Field(default='')
    instructions: str = Field(default='')

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
    def analyze(self, experience: ShortTermMemory, memory: Optional[LongTermMemory] = None) -> ShortTermMemory:
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
        experience = self.analyze(experience, memory)
        experience = self.improve(experience)
        print('Done!')
        return experience


class Skill(BaseSkill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """
    # TODO: idk if we need this...
    prediction_field: str = 'predictions'

    # _previous_instructions: Optional[List[str]] = []

    def _call(self, input: InternalDataFrame, runtime: Runtime) -> InternalDataFrame:
        extra_fields = self.model_dump(exclude={'name', 'description', 'prompt_template', 'instructions'})
        runtime_outputs = runtime.process_batch(
            input,
            prompt_template=self.prompt_template,
            instructions=self.instructions,
            extra_fields=extra_fields
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

    def analyze(self, experience: ShortTermMemory, memory: Optional[LongTermMemory] = None) -> ShortTermMemory:
        errors = experience.evaluations[~experience.evaluations[f'{self.prediction_field}_match']]
        accuracy = experience.evaluations[f'{self.prediction_field}_match'].mean()
        updated_experience = experience.model_copy()
        updated_experience.errors = errors
        updated_experience.accuracy = accuracy
        return updated_experience

    def improve(self, experience: ShortTermMemory) -> ShortTermMemory:
        errors = experience.errors
        num_samples = min(3, errors.shape[0])
        gt_column_name = experience.dataset.ground_truth_column
        errors_list = errors.sample(n=num_samples).apply(
            lambda r: f'INPUT: {r.drop([self.prediction_field, gt_column_name]).to_json()}\n'
                      f'PREDICTED OUTPUT: {r[self.prediction_field]}\n'
                      f'EXPECTED OUTPUT: {r[gt_column_name]}', axis=1)

        errors_str = "\n".join(errors_list.tolist())
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": '''\
Act as an 'Instruction Tuner' for the LLM. You will be given the inputs:

- The [CURRENT INSTRUCTION] used to guide the LLM's classification, including specific examples with ground truth labels.
- [CURRENT ERRORS] that emerged when this instruction was applied to a dataset.

The current errors are presented in the following format:
INPUT: [input text]
PREDICTED OUTPUT: [predicted label]
EXPECTED OUTPUT: [ground truth label]

Carefully analyze these errors and craft a revised concise instruction for the LLM to fit the expected outputs. \
Include 2-3 examples at the end of your response to demonstrate how the new instruction would be applied. \
Use the following format for your examples:

Input: [input text]
Output: [expected output label]

Use specific error examples and generalize them to address any observed errors that may occur in the future.
Deliver your response as the refined instruction.'''},
        {'role': 'user', 'content': f'''\
CURRENT INSTRUCTION: {self.instructions}
CURRENT ERRORS:

{errors_str}

New refined instruction:
        '''}])

        updated_instructions = response['choices'][0]['message']['content']

        updated_experience = experience.model_copy()
        updated_experience.initial_instructions = self.instructions
        updated_experience.updated_instructions = updated_instructions
        return updated_experience
