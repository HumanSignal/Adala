import openai

from .base import Skill, Experience, LongTermMemory
from adala.datasets.base import Dataset


class LLMExperience(Experience):
    errors: Dataset


class LLMSkill(Skill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """
    model_name: str = 'gpt-3.5-turbo-instruct'
    temperature: float = 0
    prompt_template: str = '{instruction}\n\nInput: {input}\nOutput:\n'
    verbose: bool = False

    def apply(self, dataset: Dataset) -> Dataset:
        completions = []
        predictions_column_name = self.name
        for batch in dataset.template_string_batches(
            template=self.prompt_template, instruction=self.instruction,
            # this is the current OpenAI limit
            batch_size=20
        ):
            result = openai.Completion.create(model=self.model_name, prompt=batch)
            completions.extend({predictions_column_name: c['text'] for c in result['choices']})

        predictions = dataset.make_new_with_index(completions)
        return predictions

    def evaluate(self, dataset: Dataset, predictions: Dataset) -> Dataset:
        predictions_with_ground_truth = predictions.assign(dataset.get_ground_truth())
        predictions_with_ground_truth.assign_columns_match(
            column_a=self.name,
            column_b=dataset.ground_truth_column,
            inplace=True,
            output_column_name=f'{self.name}_match'
        )
        return predictions_with_ground_truth

    def analyze(
        self, original_dataset: Dataset, evaluation: Dataset, memory: LongTermMemory
    ) -> Experience:
        match_column = f'{self.name}_match'
        stats = evaluation.get_column_stats(match_column)
        if self.verbose:
            print(f'LLM Skill {self.name} stats: {stats}')
        errors = evaluation.simple_select(match_column, False)
        return LLMExperience(errors=errors)

    def improve(self, original_dataset: Dataset, experience: LLMExperience) -> None:
        errors = LLMExperience.errors
        num_samples = min(3, len(errors))
        few_errors = errors.sample(n=num_samples)
        few_shot_list = few_errors.apply_template(f'INPUT: {{text}} -> {{{self.name}}} '
                                                  f'(Must be {{{original_dataset.ground_truth_column}}}').tolist()
        errors_str = "\n".join(few_shot_list)
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

