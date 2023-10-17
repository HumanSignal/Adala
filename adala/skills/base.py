import openai
import pandas as pd

from pydantic import BaseModel
from typing import List, Optional, Any
from abc import ABC, abstractmethod
from pydantic import Field, field_serializer

from typing import Optional
from adala.datasets.base import InternalDataFrame, InternalDataFrame_encoder
from adala.runtimes.base import LLMRuntime

from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import Memory, Experience


class BaseSkill(BaseModel, ABC):
    name: str
    instructions: str
    description: Optional[str]

    _previous_instructions: Optional[List[str]] = []

    @abstractmethod
    def apply(self, dataset: Dataset, runtime: Runtime) -> Dataset:
        """
        Apply skill to dataset and return new dataset with skill results (predictions)
        """

    @abstractmethod
    def evaluate(self, original_dataset: Dataset, predictions: Dataset) -> Dataset:
        """
        Test predictions and return new Dataset with evaluation results - e.g. error examples
        """

    @abstractmethod
    def analyze(
        self, original_dataset: Dataset, evaluation: Dataset, memory: Memory
    ) -> Experience:
        """
        Analyze results and return observed experience - it will be stored in long term memory
        """

    @abstractmethod
    def improve(self, dataset: Dataset, experience: Experience) -> None:
        """
        Improve current skill state based on current experience
        """

    def learn(self, dataset: Dataset, runtime: Runtime, memory: Memory) -> Experience:
        """
        Apply, validate, analyze and optimize skill.
        """
        predictions = self.apply(dataset, runtime=runtime)
        annotated_dataset = self.evaluate(dataset, predictions)
        experience = self.analyze(dataset, annotated_dataset, memory)
        self.improve(dataset, experience)
        return experience


class LLMExperience(Experience):
    errors: InternalDataFrame
    accuracy: float  # TODO: implement moving average

    class Config:
        arbitrary_types_allowed = True

    @field_serializer('errors')
    def serialize_dt(self, errors: InternalDataFrame):
        return str(InternalDataFrame_encoder(errors))


class Skill(BaseSkill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """
    prompt_template: str
    labels: Optional[List[str]] = None

    def apply(self, dataset: Dataset, runtime: LLMRuntime) -> InternalDataFrame:
        predictions = []

        extra_fields = {'instructions': self.instructions}
        if self.labels:
            extra_fields['labels'] = self.labels

        for batch in dataset.batch_iterator(
            # this is the current OpenAI limit
            batch_size=20
        ):
            runtime_outputs = runtime.process_batch(
                batch,
                prompt_template=self.prompt_template,
                extra_fields=extra_fields,
            )
            predictions.extend(runtime_outputs)

        predictions = dataset.make_new_with_index(predictions)
        return predictions

    def evaluate(self, dataset: Dataset, predictions: InternalDataFrame) -> InternalDataFrame:
        gt = dataset.get_ground_truth()
        pred = predictions.loc[gt.index]
        pred = pred[pred.notna()]

        # TODO: implement more sophisticated evaluation beyond simple equality
        match = pred[self.name] == gt[dataset.ground_truth_column]
        pred[f'{self.name}_match'] = match
        return pd.concat([gt, pred], axis=1)

    def analyze(
        self, original_dataset: Dataset, evaluation: InternalDataFrame, memory: Optional[Memory] = None
    ) -> Experience:
        errors = evaluation[~evaluation[f'{self.name}_match']]
        accuracy = evaluation[f'{self.name}_match'].mean()
        return LLMExperience(errors=errors, accuracy=accuracy)

    def improve(self, original_dataset: Dataset, experience: LLMExperience) -> None:
        errors = experience.errors
        num_samples = min(3, errors.shape[0])
        pred_column_name = self.name
        gt_column_name = original_dataset.ground_truth_column
        errors_list = errors.sample(n=num_samples).apply(
            lambda r: f'INPUT: {r.drop([pred_column_name, gt_column_name]).to_json()}\n'
                      f'PREDICTED OUTPUT: {r[pred_column_name]}\n'
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

        new_instructions = response['choices'][0]['message']['content']
        self._previous_instructions.append(self.instructions)
        self.instructions = new_instructions
