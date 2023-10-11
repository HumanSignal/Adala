import json
import difflib
import pandas as pd
import logging

from uuid import uuid4
from copy import deepcopy
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain import PromptTemplate, OpenAI, LLMChain

logger = logging.getLogger(__name__)


class LLMPredictor:
    # TODO: use RAG in 'Example' section
    PROMPT_TEMPLATE = '''\
Classify the following JSON record [RECORD] based on these instructions [INSTRUCTIONS] and choose from the provided labels [LABELS].

Example:
INSTRUCTIONS: "Identify if the statement is about nature."
RECORD: {{"text": "The sky is blue."}}
LABELS: [Yes, No]
ANSWER:
Yes

INSTRUCTIONS: "{instructions}"
RECORD: {record}
LABELS: {labels}
ANSWER:
'''

    def __init__(self):
        self.llm = OpenAI(model_name='text-davinci-003', temperature=0)
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.PROMPT_TEMPLATE),
            # verbose=True
        )

    def match_labels(self, response: str, original_labels: List[str]):
        scores = list(map(lambda l: difflib.SequenceMatcher(None, response, l).ratio(), original_labels))
        return original_labels[scores.index(max(scores))]

    def __call__(self, row: pd.Series, instructions: str, labels: List):
        row_dict = row.to_dict()
        prediction = self.llm_chain.predict(
            record=json.dumps(row_dict),
            instructions=instructions,
            labels=str(labels)
        )
        safe_prediction = self.match_labels(prediction, labels)
        return safe_prediction

    def predict(self, df: pd.DataFrame, instructions, labels, prediction_column='predictions') -> pd.DataFrame:
        predictions = df.apply(
            func=self,
            axis=1,
            instructions=instructions,
            labels=labels,
        )
        return df.assign(**{prediction_column: predictions})


def predict(instruction, df, labels):
    predictions = df.apply(
        func=LLMPredictor(),
        axis=1,
        instructions=instruction,
        labels=labels,
    )
    df_pred = df.assign(predictions=predictions)
    return df_pred


def calc_fitness(records, df, labels, ground_truth_column, sample_size=5, top_n=5):
    df = df.sample(n=sample_size, axis=0)
    output_records = deepcopy(records)
    for record in output_records:
        df_pred = predict(record['instruction'], df.drop(columns=[ground_truth_column]), labels)
        current_matches = (df_pred['predictions'] == df[ground_truth_column]).sum()
        examples_seen = record['examples_seen']
        total_examples_seen = examples_seen + sample_size
        # iterative formula for calculating accuracy
        record['accuracy'] = (examples_seen * record['accuracy'] + current_matches) / total_examples_seen
        record['examples_seen'] = total_examples_seen
        record['errors'] = df_pred[df_pred['predictions'] != df[ground_truth_column]].to_json(orient='records')

    sorted_results = sorted(output_records, key=lambda x: x['accuracy'], reverse=True)
    best_results = sorted_results[:top_n]
    return best_results


def adapt(current_instruction, errors, labels):
    llm = ChatOpenAI(model_name='gpt-4', temperature=0.5)
    system_message_prompt = SystemMessagePromptTemplate.from_template('''\
Act as an 'Instruction Tuner' for the LLM. You will be given the inputs:

- The [CURRENT INSTRUCTION] used to guide the LLM's classification
- Target set of [LABELS] for the dataset in question.
- [CURRENT ERRORS] that emerged when this instruction was applied to a dataset.

The ERRORS presented in JSON format, which contain the ground_truth label, \
the predictions label, and the input data in one or more columns. \
Here's an example format for the errors:

```json
[{{"ground_truth":"...","predictions":"...", "input_text": "...", "other": "data", ...}}, {{"ground_truth":"...","predictions":"...", "input_text": "...", "other": "data", ...}}, ...]
```

Analyze these inputs and craft a revised instruction for the LLM, aiming to enhance classification accuracy for the dataset in question. Deliver your response as the refined instruction.
''')
    human_message_prompt = HumanMessagePromptTemplate.from_template('''\
CURRENT INSTRUCTION: "{instruction}"
LABELS: {labels}
CURRENT ERRORS: {errors}

New refined instruction:
''')
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    new_instructions = chain.run(
        instruction=current_instruction,
        errors=errors,
        labels=labels
    )
    return new_instructions


def mutate(current_instruction):
    llm = ChatOpenAI(model_name='gpt-4', temperature=0.5)
    system_message_prompt = SystemMessagePromptTemplate.from_template('''\
Assume the role of an 'Instruction Optimizer' for the LLM.
Examine the [CURRENT INSTRUCTION] provided. \
Your task is to infuse it with common sense knowledge while keeping alterations minimal. \
Deliver a concise, clear, and improved version.''')
    human_message_prompt = HumanMessagePromptTemplate.from_template('''\
CURRENT INSTRUCTION: "{instruction}"

New instruction:
''')
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    new_instructions = chain.run(instruction=current_instruction)
    return new_instructions


def optimize(
    df: pd.DataFrame,
    ground_truth_column: str,
    initial_instructions: List,
    num_generations=10,
    top_instructions=5,
    validation_sample_size=5,
):
    records = [
        {
            'instruction': instruction,
            'errors': '[]',
            'accuracy': 0,
            # 'variance': 0,
            'examples_seen': 0,
            'id': uuid4().hex[:4]
        }
        for instruction in initial_instructions
    ]
    labels = df[ground_truth_column].unique().tolist()
    for generation in range(num_generations):
        # calculate fitness value and corresponding errors
        logger.info(f'Calculating fitness for {len(records)} instructions')
        records = calc_fitness(
            records=records,
            df=df,
            labels=labels,
            ground_truth_column=ground_truth_column,
            sample_size=validation_sample_size,
            top_n=top_instructions,
        )

        # mutate the best instructions with accuracy<100% based on errors
        best_results_with_errors = next((x for x in records if x['accuracy'] < 1), None)
        if not best_results_with_errors:
            # TODO: change this to a more sophisticated mutation
            logger.info(f'All instructions have 100% accuracy. Mutating the best instruction {records[0]["id"]}...')
            new_instruction = mutate(records[0]['instruction'])
        else:
            logger.info(f'Adapting the instruction {best_results_with_errors["id"]}...')
            new_instruction = adapt(best_results_with_errors['instruction'], best_results_with_errors['errors'], labels)

        # save only the best instructions and the new one
        records = records + [{
            'instruction': new_instruction,
            'errors': '[]',
            'accuracy': 0,
            # 'variance': 0,
            'examples_seen': 0,
            'id': uuid4().hex[:4]
        }]

        logger.info(
            f'Results of {generation} generation:\n'
            f'{pd.DataFrame.from_records(records)[["id", "instruction", "accuracy", "examples_seen"]]}')

    # calculate fitness on final results
    fitness = calc_fitness(records, df, labels, ground_truth_column, validation_sample_size, top_instructions)
    logger.info(
        f'Final results:\n{pd.DataFrame.from_records(fitness)[["id", "instruction", "accuracy", "examples_seen"]]}')
    return fitness


def generate_instructions(
    df: pd.DataFrame,
    ground_truth_column: str,
    initial_instructions: Optional[List] = None,
    num_generations=10,
    top_instructions=5,
    validation_sample_size=5,
):
    """
    Generates instructions for the LLM to classify the data in the given dataframe.
    :param df:
    :param ground_truth_column:
    :param initial_instructions:
    :param num_generations:
    :param top_instructions:
    :param validation_sample_size:
    :return:
    """
    results = optimize(
        df=df,
        ground_truth_column=ground_truth_column,
        initial_instructions=initial_instructions or [''],
        num_generations=num_generations,
        top_instructions=top_instructions,
        validation_sample_size=validation_sample_size,
    )
    return results[0]['instruction']
