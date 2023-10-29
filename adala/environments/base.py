from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable, Dict

from adala.utils.internal_data import InternalDataFrame, InternalSeries, InternalDataFrameConcat
from adala.skills.base import BaseSkill
from adala.skills.skillset import SkillSet
from adala.datasets import Dataset, DataFrameDataset


class GroundTruthSignal(BaseModel):
    match: InternalDataFrame
    errors: Optional[Dict[str, InternalDataFrame]] = None

    def get_accuracy(self) -> InternalSeries:
        return self.match.mean()

    def get_errors(self, skill_name: str) -> InternalDataFrame:
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
    """Abstract base class for environments.

    The environment provides a mechanism to obtain ground truth information from raw data and predictions, 
    and also facilitates comparison of ground truth with predictions.

    Attributes:
        Config (class): Configuration for the environment class, allows arbitrary types.
    """
        
    @abstractmethod
    def request_feedback(self, skill_set: SkillSet, predictions: InternalDataFrame):
        """Request user feedback using predictions and update internal ground truth set."""

    @abstractmethod
    def compare_to_ground_truth(self, skill_set: SkillSet, predictions: InternalDataFrame) -> GroundTruthSignal:
        """Compare predictions with ground truth and return the results."""

    @abstractmethod
    def as_dataset(self) -> Dataset:
        """Convert the environment to a dataset."""

    @abstractmethod
    def save(self):
        """Persist the state of the environment."""

    @abstractmethod
    def restore(self):
        """Retrieve and set the state of the environment."""

    class Config:
        arbitrary_types_allowed = True


class BasicEnvironment(Environment):
    """Basic environment implementation.

    This environment assumes the ground truth is provided explicitly with the input data.
    For comparison with ground truth, exact matching is used.

    Attributes:
        ground_truth_dataset (DataFrameDataset): Dataset containing the ground truth data.
                                                 Defaults to an empty DataFrameDataset.
        ground_truth_column (str): Name of the column containing ground truth in the dataset.
                                   Defaults to 'ground_truth'.

    """
    
    ground_truth_dataset: Union[InternalDataFrame, DataFrameDataset] = Field(default_factory=DataFrameDataset)
    ground_truth_columns: Dict[str, str]

    @field_validator('ground_truth_dataset')
    def _validate_ground_truth_dataset(cls, v):
        if isinstance(v, InternalDataFrame):
            return DataFrameDataset(df=v)
        return v

    def request_feedback(self, skill: BaseSkill, predictions: InternalDataFrame):
        """In the BasicEnvironment, ground truth is already provided with the input data."""

    def compare_to_ground_truth(self, skill_set: SkillSet, predictions: InternalDataFrame) -> GroundTruthSignal:
        """Compare the predictions with the ground truth using exact matching.

        Args:
            skill_set (SkillSet): The skill set being evaluated.
            predictions (InternalDataFrame): The predictions to compare with ground truth.
        Returns:
            GroundTruthSignal: The ground truth signal.
        """

        ground_truth_match = InternalDataFrame()
        errors = {}
        for skill_id, skill in skill_set.skills.items():
            gt_column = self.ground_truth_columns[skill.name]
            gt = self.ground_truth_dataset.df[gt_column]
            pred = predictions[skill.name]
            # from ground truth dataset, select only the rows that are in the predictions
            gt, pred = gt.align(pred)
            # compare ground truth with predictions
            # TODO: we can customize the matching function here beyond exact matching
            gt_pred_match = (gt == pred)[gt.notnull() & pred.notnull()]
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
        """
        
        return self.ground_truth_dataset

    def save(self):
        """Save method for BasicEnvironment. Not implemented."""
        
        raise NotImplementedError

    def restore(self):
        """Restore method for BasicEnvironment. Not implemented."""
        
        raise NotImplementedError
