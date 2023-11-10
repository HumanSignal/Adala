from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable, Dict
from collections import defaultdict

from adala.utils.internal_data import InternalDataFrame, InternalSeries, InternalDataFrameConcat
from adala.utils.matching import fuzzy_match
from adala.skills.base import BaseSkill
from adala.skills.skillset import SkillSet
from adala.datasets import Dataset, DataFrameDataset


class GroundTruthSignal(BaseModel):
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
        errors (Optional[Dict[str, InternalDataFrame]]): A dictionary mapping skill names to DataFrames
            containing the errors between predictions and ground truth. Default is None.
            Each DataFrame has two columns ["predictions", "user-defined ground truth name"].
            User defined ground truth name is taken from Environment
            Example:
            ```json
                {
                    "skill_1": InternalDataFrame({
                        "predictions": ['a', 'b', 'c'],
                        "my_gr_truth": ['a', 'a', 'c']
                    }, index=[0, 1, 2])
                }
            ```
    """
    
    match: InternalDataFrame = Field(default_factory=InternalDataFrame)
    errors: Optional[Dict[str, InternalSeries]] = None

    def get_accuracy(self) -> InternalSeries:
        """
        Calculate the accuracy of predictions as the mean of matches.

        Returns:
            InternalSeries: A series representing the accuracy of predictions.

        Examples:
            

        """
        return self.match.mean()

    def get_errors(self, skill_output: str) -> InternalSeries:
        """
        Retrieve the errors associated with a particular skill.

        Args:
            skill_name (str): The name of the skill to retrieve errors for.

        Returns:
            InternalSeries: A series representing the errors of predictions for the given skill.
            Index is prediction index, value is ground truth.
        Raises:
            AssertionError: If the error DataFrame does not have exactly two columns.
        """
        
        return self.errors[skill_output]

    def __rich__(self):
        text = '[bold blue]Ground Truth Signal:[/bold blue]\n\n'
        text += f'\n[bold]Match[/bold]\n{self.match}'
        if self.errors is not None:
            for skill_name, errors in self.errors.items():
                text += f'\n[bold]Errors for {skill_name}[/bold]\n{errors}'
        return text

    class Config:
        arbitrary_types_allowed = True


class Environment(BaseModel, ABC):
    """
    An abstract base class that defines the structure and required methods for an environment
    in which machine learning models operate and are evaluated against ground truth data.

    Subclasses should implement methods to handle feedback requests, comparison to ground truth,
    dataset conversion, and state persistence.
    """
    ground_truth_columns: Optional[Dict[str, str]] = None
    matching_function: str = 'exact'
    matching_threshold: float = 0.8

    @abstractmethod
    def get_data_batch(self) -> InternalDataFrame:
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
    def get_ground_truth(self, predictions: InternalDataFrame) -> InternalDataFrame:
        """
        Get ground truth data for the predictions.

        Args:
            predictions (InternalDataFrame): The predictions to compare with the ground truth.

        Returns:
            InternalDataFrame: The ground truth data for the predictions.
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

    def compare_to_ground_truth(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
    ) -> GroundTruthSignal:
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

        errors = {}
        ground_truth_dataset = self.get_ground_truth(predictions=predictions)
        if ground_truth_dataset.empty:
            raise ValueError('Ground truth dataset is empty. Run `request_feedback()` first.')

        pred_columns = list(skills.get_skill_outputs())

        skill_match = {}
        for pred_column in pred_columns:
            if not self.ground_truth_columns:
                gt_column = pred_column
            else:
                gt_column = self.ground_truth_columns[pred_column]
            gt = ground_truth_dataset[gt_column]
            pred = predictions[pred_column]
            # from ground truth dataset, select only the rows that are in the predictions
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

            # for values with True, we assume them equal to predictions
            gt_pred_match[gt == True] = True
            gt_pred_match[gt == False] = False

            error_index = gt_pred_match[~gt_pred_match].index
            # concatenate errors - dataframe with two columns: predictions and ground truth
            errors[pred_column] = gt[error_index]
            # concatenate matching columns
            skill_match[pred_column] = gt_pred_match
        match = InternalDataFrame(skill_match).reindex(predictions.index)

        return GroundTruthSignal(match=match, errors=errors)

    class Config:
        arbitrary_types_allowed = True


class StaticEnvironment(Environment):
    """
    Static environment that initializes everything from the dataframe
    and doesn't not require requesting feedback to create the ground truth.
    """    
    df: InternalDataFrame = None

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

    def get_ground_truth(self, predictions: InternalDataFrame) -> InternalDataFrame:
        """
        Get the ground truth dataset.
        """
        return self.df

    def get_data_batch(self) -> InternalDataFrame:
        """
        Return the dataset containing the ground truth data.

        Returns:
            Dataset: The ground truth dataset as a DataFrameDataset.
        """
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
