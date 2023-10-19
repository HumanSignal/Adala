import logging
import pandas as pd

from dataclasses import dataclass
from typing import List
from uuid import uuid4
from copy import deepcopy
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from adala.labelers.base import LLMLabeler

logger = logging.getLogger(__name__)


def calc_fitness(labeler: LLMLabeler, records, df, labels, ground_truth_column, sample_size=5, top_n=5):
    df = df.sample(n=sample_size, axis=0)
    output_records = deepcopy(records)
    for i, record in enumerate(output_records):
        df_pred = labeler.label(
            df=df.drop(columns=[ground_truth_column] + [r['id'] for r in output_records[:i]]),
            instruction=record['instruction'],
            labels=labels
        )
        df_pred = pd.concat((df_pred, df[[ground_truth_column]]), axis=1)
        current_matches = (df_pred['predictions'] == df_pred[ground_truth_column]).sum()
        current_mismatches = (df_pred['predictions'] != df_pred[ground_truth_column]).sum()
        examples_seen = record['examples_seen']
        total_examples_seen = examples_seen + sample_size
        # iterative formula for calculating accuracy
        record['accuracy'] = (examples_seen * record['accuracy'] + current_matches) / total_examples_seen
        record['examples_seen'] = total_examples_seen
        record['errors'] = '\n'.join(df_pred[df_pred['predictions'] != df_pred[ground_truth_column]].apply(
            lambda r: f'INPUT: {r.drop(["predictions", ground_truth_column]).to_json()}\n'
                      f'PREDICTED OUTPUT: {r["predictions"]}\n'
                      f'EXPECTED OUTPUT: {r[ground_truth_column]}', axis=1))
        record['mismatches'] = current_mismatches

        df[record['id']] = df_pred['predictions']

    logger.info(f'Result df:\n{df}')

    sorted_results = sorted(output_records, key=lambda x: x['accuracy'], reverse=True)
    best_results = sorted_results[:top_n]
    return best_results


def regularize(instruction: str):
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
    new_instructions = chain.run(instruction=instruction)
    return new_instructions


def adapt(current_instruction, errors, labels):
    llm = ChatOpenAI(model_name='gpt-4', temperature=0.5)
    system_message_prompt = SystemMessagePromptTemplate.from_template('''\
Act as an 'Instruction Tuner' for the LLM. You will be given the inputs:

- The [CURRENT INSTRUCTION] used to guide the LLM's classification, including specific examples with ground truth labels.
- Target set of [LABELS] for the dataset in question.
- [CURRENT ERRORS] that emerged when this instruction was applied to a dataset.

The current errors are presented in the following format:
INPUT: [input text]
PREDICTED OUTPUT: [predicted label]
EXPECTED OUTPUT: [ground truth label]

Carefully analyze these errors and craft a revised instruction for the LLM to fit the expected outputs. \
Include 2-3 examples at the end of your response to demonstrate how the new instruction would be applied. \
Use the following format for your examples:

INPUT: [input text]
OUTPUT: [expected output label]

Use specific error examples and generalize them to address any observed errors that may occur in the future. 
Deliver your response as the refined instruction.''')
#Analyze these inputs and craft a revised instruction for the LLM, aiming to enhance classification accuracy for the dataset in question. Deliver your response as the refined instruction.
#''')
    human_message_prompt = HumanMessagePromptTemplate.from_template('''\
CURRENT INSTRUCTION: {instruction}
LABELS: {labels}
CURRENT ERRORS:

{errors}

New refined instruction:
''')
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=False)
    new_instructions = chain.run(
        instruction=current_instruction,
        errors=errors,
        labels=labels
    )
    return new_instructions


@dataclass
class GenerateInstructionResult:
    """Result of the generate_instruction()"""
    best_instruction: str
    benchmark: pd.DataFrame
    labels: List[str]


def generate_instruction(
    labeler: LLMLabeler,
    df: pd.DataFrame,
    ground_truth_column: str = 'ground_truth',
    initial_instructions: List = None,
    max_iterations: int = 10,
    top_instructions: int = 3,
    validation_sample_size: int = 5,
    human_in_the_loop: bool = False,
    label_studio_project_id: int = None,
    label_studio_api_token: str = None,
    label_studio_host: str = None,
) -> GenerateInstructionResult:
    """Optimize the instruction for the LLM."""

    initial_instructions = initial_instructions or ['']
    df = df.dropna(subset=[ground_truth_column])
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
    # labels = None
    for iteration in range(max_iterations):
        # calculate fitness value and corresponding errors
        logger.info(f'Calculating fitness for {len(records)} instructions')
        records = calc_fitness(
            labeler=labeler,
            records=records,
            df=df,
            labels=labels,
            ground_truth_column=ground_truth_column,
            sample_size=validation_sample_size,
            top_n=top_instructions,
        )

        logger.info(
            f'Results of {iteration} iteration(s):\n'
            f'{pd.DataFrame.from_records(records)[["id", "instruction", "accuracy", "examples_seen", "mismatches"]]}')

        # mutate the best instructions with accuracy<100% based on errors
        best_results_with_errors = next((x for x in records if x['mismatches'] > 0), None)
        if not best_results_with_errors:
            # TODO: change this to a more sophisticated mutation
            logger.info(f'All instructions have 100% accuracy. Mutating the best instruction {records[0]["id"]}...')
            new_instruction = regularize(records[0]['instruction'])
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

    # calculate fitness on final results
    records = calc_fitness(
        labeler=labeler,
        records=records,
        df=df,
        labels=labels,
        ground_truth_column=ground_truth_column,
        sample_size=len(df),
        top_n=top_instructions,
    )
    benchmark_table = pd.DataFrame.from_records(records)[["id", "instruction", "accuracy", "examples_seen", "mismatches"]]
    logger.info(f'Final results:\n{benchmark_table}')

    return GenerateInstructionResult(
        best_instruction=records[0]['instruction'],
        benchmark=benchmark_table,
        labels=labels
    )

