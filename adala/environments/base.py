from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable, Dict

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
        errors (Optional[Dict[str, InternalDataFrame]]): A dictionary mapping skill names to DataFrames
            containing the errors between predictions and ground truth. Default is None.
    """
    
    match: InternalDataFrame
    errors: Optional[Dict[str, InternalDataFrame]] = None

    def get_accuracy(self) -> InternalSeries:
        """
        Calculate the accuracy of predictions as the mean of matches.

        Returns:
            InternalSeries: A series representing the accuracy of predictions.
        """
        
        return self.match.mean()

    def get_errors(self, skill_name: str) -> InternalDataFrame:
        """
        Retrieve the errors associated with a particular skill.

        Args:
            skill_name (str): The name of the skill to retrieve errors for.

        Returns:
            InternalDataFrame: A DataFrame with two columns ["predictions", "ground_truth name"]
            representing the errors.

        Raises:
            AssertionError: If the error DataFrame does not have exactly two columns.
        """
        
        errors = self.errors[skill_name]
        assert len(errors.columns) == 2  # ["predictions", "ground_truth name"]
        return errors

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
            
    @abstractmethod
    def request_feedback(self, skill_set: SkillSet, predictions: InternalDataFrame):
        """
        Abstract method to request user feedback on the predictions made by the model.

        Args:
            skill_set (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions made by the skills/models.
        """

    @abstractmethod
    def compare_to_ground_truth(self, skill_set: SkillSet, predictions: InternalDataFrame) -> GroundTruthSignal:
        """
        Abstract method to compare predictions with ground truth data.

        Args:
            skill_set (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions made by the skills/models.

        Returns:
            GroundTruthSignal: An instance of GroundTruthSignal containing the comparison results.
        """

    @abstractmethod
    def as_dataset(self) -> Dataset:
        """
        Abstract method to convert the environment's state into a dataset.

        Returns:
            Dataset: A dataset representing the environment's state.
        """

    @abstractmethod
    def save(self):
        """
        Abstract method to persist the current state of the environment.
        """

    @abstractmethod
    def restore(self):
        """
        Abstract method to restore the environment's state from persisted data.
        """

    class Config:
        arbitrary_types_allowed = True


class BasicEnvironment(Environment):
    """
    A concrete implementation of the Environment abstract base class,
    assuming the ground truth is provided and comparison is based on exact or fuzzy matching.

    Attributes:
        ground_truth_dataset (Union[InternalDataFrame, DataFrameDataset]): Dataset containing
            the ground truth data, defaulting to an empty DataFrameDataset.
        ground_truth_columns (Dict[str, str]): A dictionary mapping skill names to their corresponding
            ground truth columns in the dataset.
        matching_function (str): The name of the matching function to use, defaults to 'exact'.
        matching_threshold (float): The threshold for fuzzy matching, defaults to 0.8.
    """    
    
    ground_truth_dataset: Union[InternalDataFrame, DataFrameDataset] = Field(default_factory=DataFrameDataset)
    ground_truth_columns: Dict[str, str]
    matching_function: str = 'exact'
    matching_threshold: float = 0.8

    @field_validator('ground_truth_dataset')
    def _validate_ground_truth_dataset(cls, v):
        """
        Validate the ground_truth_dataset field to ensure it is converted to DataFrameDataset if needed.

        Args:
            v: The value to validate.

        Returns:
            The validated value, possibly converted to DataFrameDataset.

        Raises:
            ValidationError: If the validation fails.
        """
        
        if isinstance(v, InternalDataFrame):
            return DataFrameDataset(df=v)
        return v

    def request_feedback(self, skill: BaseSkill, predictions: InternalDataFrame):
        """
        In the BasicEnvironment, this method is a placeholder as ground truth is already provided with the input data.

        Args:
            skill (BaseSkill): The skill being evaluated.
            predictions (InternalDataFrame): The predictions to be reviewed.
        """        

    def compare_to_ground_truth(self, skill_set: SkillSet, predictions: InternalDataFrame) -> GroundTruthSignal:
        """Compare the predictions with the ground truth using exact matching.

        Args:
            skill_set (SkillSet): The skill set being evaluated.
            predictions (InternalDataFrame): The predictions to compare with ground truth.
        Returns:
            GroundTruthSignal: The ground truth signal.
        """"""
        Compare the predictions with the ground truth using the specified matching function.

        Args:
            skill_set (SkillSet): The skill set being evaluated.
            predictions (InternalDataFrame): The predictions to compare with the ground truth.

        Returns:
            GroundTruthSignal: The resulting ground truth signal, with matches and errors detailed.

        Raises:
            NotImplementedError: If the matching_function is unknown.
        """

        ground_truth_match = InternalDataFrame()
        errors = {}
        for skill_id, skill in skill_set.skills.items():
            gt_column = self.ground_truth_columns[skill.name]
            gt = self.ground_truth_dataset.df[gt_column]
            pred = predictions[skill.name]
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

            error_index = gt_pred_match[~gt_pred_match].index
            # concatenate errors - dataframe with two columns: predictions and ground truth
            errors[skill.name] = InternalDataFrameConcat([pred[error_index], gt[error_index]], axis=1)
            errors[skill.name].columns = ["predictions", gt_column]
            # concatenate matching columns
            ground_truth_match = InternalDataFrameConcat([
                # previous skills' ground truth matches
                ground_truth_match,
                # current skill's ground truth match
                gt_pred_match.rename(skill.name),
            ], axis=1)

        return GroundTruthSignal(
            match=ground_truth_match.reindex(predictions.index),
            errors=errors
        )

    def as_dataset(self) -> Dataset:
        """Return the ground truth dataset.

        Returns:
            Dataset: The dataset containing ground truth data.
        """"""
        Return the dataset containing the ground truth data.

        Returns:
            Dataset: The ground truth dataset as a DataFrameDataset.
        """
        
        return self.ground_truth_dataset

    def save(self):
        """Save method for BasicEnvironment. Not implemented.""""""
        Save the current state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """
        
        raise NotImplementedError

    def restore(self):
        """Restore method for BasicEnvironment. Not implemented.""""""
        Restore the state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """
        
        raise NotImplementedError
