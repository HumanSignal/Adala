import logging
import pandas as pd

from typing import Optional, Dict
from label_studio_sdk.utils import get_or_create_project

from .analyst import Analyst
from .engineer import Engineer
from .labeler import Labeler

logger = logging.getLogger(__name__)


class Adala:

    def __init__(self):
        self.labeler = Labeler()
        self.analyst = Analyst()
        self.engineer = Engineer()

    def run(
        self,
        df: pd.DataFrame,
        initial_instructions: str,
        label_studio_project_id: int,
        validation_sample_size: Optional[int] = 5,
        max_iterations: Optional[int] = 10,
        max_accuracy: Optional[float] = 0.9
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the ADALA on the input data frame and give back output with instructions and accuracies
        :param df:
        :param initial_instructions:
        :param label_studio_project_id:
        :param validation_sample_size:
        :param max_iterations:
        :param max_accuracy:
        :return:
        """

        project = get_or_create_project(project_id=label_studio_project_id)
        labels = next(iter(project.parsed_label_config.values()))['labels']

        logger.info(
            f'Connected to project: {project.title} (ID={project.id})\n'
            f'Target labels: {labels}'
        )

        current_instructions = initial_instructions

        df = df.copy()
        prev_df_val = project.get_dataframe()
        logger.debug(f'Retrieved dataframe from project:\n{prev_df_val}')

        history = []

        for iteration in range(max_iterations):
            df_val = df.sample(n=validation_sample_size, axis=0)
            df.drop(df_val.index)

            predictions = df_val.apply(
                func=Labeler(),
                axis=1,
                instructions=current_instructions,
                labels=labels
            )
            df_val = df_val.assign(predictions=predictions)

            df_val = project.label_dataframe(df_val, preannotated_from_fields=['predictions'])

            if not prev_df_val.empty:
                predictions = prev_df_val.apply(
                    func=Labeler(),
                    axis=1,
                    instructions=current_instructions,
                    labels=labels
                )
                prev_df_val = prev_df_val.assign(predictions=predictions)
                df_val = pd.concat([prev_df_val, df_val])

            logger.debug(f'Updated dataframe:\n{df_val}')
            prev_df_val = df_val
            accuracy_score = (df_val['predictions'] == df_val['ground_truth']).mean()
            accuracy = f'{round(100 * accuracy_score)}%'
            logger.info(f'Accuracy: {accuracy}')
            history.append({
                'iteration': iteration,
                'accuracy': accuracy,
                'instructions': current_instructions
            })

            if accuracy_score > max_accuracy:
                logger.info(f'Accuracy threshold reached: {accuracy_score} > {max_accuracy}')
                break

            observations = self.analyst(df_val)

            new_instructions = self.engineer(current_instructions, observations)

            logger.info(f'Old instructions: {current_instructions}\nNew instructions: {new_instructions}')

            current_instructions = new_instructions

        # run predictions on the rest of the dataset
        predictions = df.apply(
            func=Labeler(),
            axis=1,
            instructions=current_instructions,
            labels=labels
        )
        df = pd.concat(
            [prev_df_val, df.assign(predictions=predictions)]
        )
        return {
            'predicted_df': df,
            'history': pd.DataFrame.from_records(history)
        }


def label(
    df: pd.DataFrame,
    initial_instructions: str,
    label_studio_project_id: int,
    validation_sample_size: Optional[int] = 5,
    max_iterations: Optional[int] = 10,
    max_accuracy: Optional[float] = 0.9
) -> Dict[str, pd.DataFrame]:
    """
    Run the ADALA on the input data frame and give back output with instructions and accuracies
    :param df:
    :param initial_instructions:
    :param label_studio_project_id:
    :param validation_sample_size:
    :param max_iterations:
    :param max_accuracy:
    :return:
    """
    adala = Adala()
    return adala.run(df, initial_instructions, label_studio_project_id, validation_sample_size, max_iterations, max_accuracy)