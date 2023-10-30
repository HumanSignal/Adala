import pandas as pd
from unittest.mock import MagicMock, patch
from adala.runtimes.openai import OpenAIRuntime

from adala.agents import Agent
from adala.datasets import DataFrameDataset
from adala.environments import BasicEnvironment
from adala.skills import ClassificationSkill
from adala.utils.logs import print_dataframe


def process_record_generator(*args, **kwargs):
    # train
    for i in range(3):
        # predictions for gt comparison
        yield {'sentiment': 'Neutral' if i < 2 else 'Positive'}
        yield {'sentiment': 'Neutral' if i < 2 else 'Negative'}
        yield {'sentiment': 'Neutral'}

        # errors
        if i < 2:
            yield {'reason': 'Test reason'}
            yield {'reason': 'Test reason'}
            yield {'reason': 'Test reason'}
            yield {'reason': 'Test reason'}
            yield {'': 'Test reason'}

            # instruction generation
            yield {'new_instruction': 'Test instruction'}

    # test
    yield {'sentiment': 'Positive'}
    yield {'sentiment': 'Negative'}
    yield {'sentiment': 'Neutral'}


@patch.object(OpenAIRuntime, '_check_api_key', return_value=None)
@patch.object(OpenAIRuntime, '_check_model_availability', return_value=None)
@patch.object(OpenAIRuntime, '_process_record', side_effect=process_record_generator())
def test_classification_skill(
        mock_check_api_key,
        mock_check_model_availability,
        mock_process_record
):
    print("=> Initialize datasets ...")

    # Train dataset
    train_df = pd.DataFrame([
        ["It was the negative first impressions, and then it started working.", "Positive"],
        ["Not loud enough and doesn't turn on like it should.", "Negative"],
        ["I don't know what to say.", "Neutral"],
    ], columns=["text", "ground_truth"])

    # Test dataset
    test_df = pd.DataFrame([
        "All three broke within two months of use.",
        "The device worked for a long time, can't say anything bad.",
        "Just a random line of text.",
    ], columns=["text"])

    train_dataset = DataFrameDataset(df=train_df)
    test_dataset = DataFrameDataset(df=test_df)

    print("=> Initialize and train ADALA agent ...")
    agent = Agent(
        # connect to a dataset
        environment=BasicEnvironment(
            ground_truth_dataset=train_dataset,
            ground_truth_columns={"sentiment": "ground_truth"}
        ),
        # define a skill
        skills=ClassificationSkill(
            name='sentiment',
            instructions="Label text as subjective or objective.",
            labels=["Positive", "Negative", "Neutral"],
            input_data_field='text'
        ),
    )
    run = agent.learn(learning_iterations=3, accuracy_threshold=0.95)
    assert run.get_accuracy()['sentiment'] > 0.8

    print('\n\n=> Final instructions:')
    print('=====================')
    print(f'{agent.skills["sentiment"].instructions}')
    print('=====================')

    print('\n=> Run test ...')
    predictions = agent.run(test_dataset)
    print_dataframe(predictions)

    assert not predictions.empty
