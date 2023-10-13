import openai
import pandas as pd

from typing import Optional
from .base import Skill, Experience, LongTermMemory
from adala.datasets.base import Dataset, InternalDataFrame


class LLMExperience(Experience):
    errors: InternalDataFrame

    class Config:
        arbitrary_types_allowed = True


class LLMSkill(Skill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """
    model_name: str = 'gpt-3.5-turbo-instruct'
    temperature: float = 0
    prompt_template: str = '{instructions}\n\nInput: {input}\nOutput:\n'
    verbose: bool = False

    def apply(self, dataset: Dataset) -> InternalDataFrame:
        completions = []
        predictions_column_name = self.name
        for batch in dataset.template_string_batches(
            template=self.prompt_template, instructions=self.instructions,
            # this is the current OpenAI limit
            batch_size=20
        ):
            result = openai.Completion.create(model=self.model_name, prompt=batch)
            completions.extend({predictions_column_name: c['text']} for c in result['choices'])

        predictions = dataset.make_new_with_index(completions)
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
        self, original_dataset: Dataset, evaluation: InternalDataFrame, memory: Optional[LongTermMemory] = None
    ) -> Experience:
        errors = evaluation[~evaluation[f'{self.name}_match']]
        return LLMExperience(errors=errors)

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
