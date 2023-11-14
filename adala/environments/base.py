import pandas as pd
import numpy as np
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Optional, Dict
from adala.utils.internal_data import InternalDataFrame, InternalSeries, InternalDataFrameConcat
from adala.utils.matching import fuzzy_match
from adala.skills.skillset import SkillSet


class EnvironmentFeedback(BaseModel):
    """
        A model that represents the comparison between predictions and ground truth data,
    potentially holding information about matching results and errors per skill.

    Attributes:
        match (InternalDataFrame): A DataFrame indicating the correctness of predictions.
                                   Each row corresponds to a prediction, and each column is a boolean indicating if skill matches ground truth.
                                   Columns are named after the skill names.
                                   Indices correspond to prediction indices.
                                   Example:
                                       ```
                                        | index | skill_1 | skill_2 | skill_3 |
                                        |-------|---------|---------|---------|
                                        | 0     | True    | True    | False   |
                                        | 1     | False   | False   | False   |
                                        | 2     | True    | True    | True    |
                                        ```
        feedback (InternalDataFrame): A DataFrame that contains ground truth feedback per each skill output
    """

    match: InternalDataFrame
    feedback: InternalDataFrame

    class Config:
        arbitrary_types_allowed = True

    def get_accuracy(self) -> InternalSeries:
        """
        Calculate the accuracy of predictions as the mean of matches.

        Returns:
            InternalSeries: A series representing the accuracy of predictions.

        Examples:


        """
        return self.match.mean()

    def __rich__(self):
        text = '[bold blue]Environment Feedback:[/bold blue]\n\n'
        text += f'\n[bold]Match[/bold]\n{self.match}'
        if self.feedback is not None:
            text += f'\n[bold]Feedback[/bold]\n{self.feedback}'
        return text


class Environment(BaseModel, ABC):
    """
    An abstract base class that defines the structure and required methods for an environment
    in which machine learning models operate and are evaluated against ground truth data.

    Subclasses should implement methods to handle feedback requests, comparison to ground truth,
    dataset conversion, and state persistence.
    """

    @abstractmethod
    def get_data_batch(self, batch_size = None) -> InternalDataFrame:
        """
        Get a batch of data from data stream to be processed by the skill set.

        Returns:
            InternalDataFrame: The data batch.
        """

    @abstractmethod
    def request_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
        wait_for_feedback: Optional[bool] = False
    ):
        """
        Abstract method to request user feedback on the predictions made by the model.

        Args:
            skills (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions made by the skills.
            num_feedbacks (Optional[int], optional): The number of feedbacks to request. Defaults to all predictions
            wait_for_feedback (Optional[bool], optional): Whether to wait for feedback to be available. Defaults to False.
        """

    @abstractmethod
    def get_feedback(self, skills: SkillSet, predictions: InternalDataFrame) -> EnvironmentFeedback:
        """
        Get feedback for the predictions.

        Args:
            predictions (InternalDataFrame): The predictions to compare with the ground truth.

        Returns:
            InternalDataFrame: The feedback data for the predictions.
        """

    @abstractmethod
    def save(self):
        """
        Save the current state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    def restore(self):
        """
        Restore the state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    class Config:
        arbitrary_types_allowed = True


class StaticEnvironment(Environment):
    """
    Static environment that initializes everything from the dataframe
    and doesn't not require requesting feedback to create the ground truth.
    """    
    df: InternalDataFrame = None
    ground_truth_columns: Optional[Dict[str, str]] = None
    matching_function: str = 'fuzzy'
    matching_threshold: float = 0.9

    def request_feedback(
        self, skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
        wait_for_feedback: Optional[float] = False
    ):
        """
        In the StaticEnvironment, this method is a placeholder as ground truth is already provided with the input data.

        Args:
            skills (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions made by the skills.
            num_feedbacks (Optional[int], optional): The number of feedbacks to request. Defaults to all predictions.
            wait_for_feedback (Optional[float], optional): If True, wait for feedback to be available. Defaults to False.
        """
        pass

    def get_feedback(self, skills: SkillSet, predictions: InternalDataFrame) -> EnvironmentFeedback:
        """
        Compare the predictions with the ground truth using the specified matching function.

        Args:
            skills (SkillSet): The skill set being evaluated.
            predictions (InternalDataFrame): The predictions to compare with the ground truth.

        Returns:
            GroundTruthSignal: The resulting ground truth signal, with matches and errors detailed.

        Raises:
            NotImplementedError: If the matching_function is unknown.
        """

        pred_columns = list(skills.get_skill_outputs())
        pred_match = {}
        pred_feedback = {}

        for pred_column in pred_columns:
            if not self.ground_truth_columns:
                gt_column = pred_column
            else:
                gt_column = self.ground_truth_columns[pred_column]

            gt = self.df[gt_column]
            pred = predictions[pred_column]
            gt, pred = gt.align(pred)
            nonnull_index = gt.notnull() & pred.notnull()
            gt = gt[nonnull_index]
            pred = pred[nonnull_index]
            # compare ground truth with predictions
            if self.matching_function == 'exact':
                gt_pred_match = gt == pred
            elif self.matching_function == 'fuzzy':
                gt_pred_match = fuzzy_match(gt, pred, threshold=self.matching_threshold)
            else:
                raise NotImplementedError(f'Unknown matching function {self.matching_function}')
            pred_match[pred_column] = gt_pred_match
            # leave feedback about mismatches
            match_concat = InternalDataFrameConcat([gt_pred_match.rename('match'), gt], axis=1)
            pred_feedback[pred_column] = match_concat.apply(
                lambda row: 'Correct.' if row['match']
                else f'Incorrect. Must be equal to {row[gt_column]}' if not pd.isna(row['match']) else np.nan, axis=1)

        return EnvironmentFeedback(
            match=InternalDataFrame(pred_match).reindex(predictions.index),
            feedback=InternalDataFrame(pred_feedback).reindex(predictions.index)
        )

    def get_data_batch(self, batch_size: int = None) -> InternalDataFrame:
        """
        Return the dataset containing the ground truth data.

        Returns:
            Dataset: The ground truth dataset as a DataFrameDataset.
        """
        if batch_size is not None:
            return self.df.sample(n=batch_size)
        return self.df

    def save(self):
        """
        Save the current state of the StaticEnvironment.
        """
        raise NotImplementedError('StaticEnvironment does not support save/restore.')

    def restore(self):
        """
        Restore the state of the StaticEnvironment.
        """
        raise NotImplementedError('StaticEnvironment does not support save/restore.')
