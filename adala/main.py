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
    ) -> pd.DataFrame:
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

        for iteration in range(max_iterations):
            df_val = df.sample(n=validation_sample_size, axis=0)
            df.drop(df_val.index)

            # create ground truth
            df_val = project.label_dataframe(df_val)
            if not prev_df_val.empty:
                df_val = pd.concat([prev_df_val, df_val])

            history = []
            max_internal_iterations = 10
            for internal_iteration in range(max_internal_iterations):
                predictions = df_val.apply(
                    func=Labeler(),
                    axis=1,
                    instructions=current_instructions,
                    labels=labels
                )
                prev_df_val = df_val = df_val.assign(predictions=predictions)
                accuracy = (df_val['predictions'] == df_val['ground_truth']).mean()
                accuracy_str = f'{round(100 * accuracy)}%'
                history.append({
                    'iteration': internal_iteration,
                    'accuracy': accuracy_str,
                    'instructions': current_instructions
                })
                logger.info(f'Validation set: {df_val}')
                logger.info(f'Current state: {pd.DataFrame.from_records(history)}')
                if accuracy > max_accuracy:
                    logger.info(f'Accuracy threshold reached: {accuracy} > {max_accuracy}')
                    break
                if len(history) >= 3 and (history[-1]['accuracy'] == history[-2]['accuracy'] == history[-3]['accuracy']):
                    logger.info(f'Accuracy is not improving, trying to collect more data...')
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
        df = df.assign(predictions=predictions)
        df = pd.concat([prev_df_val, df])
        return df


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