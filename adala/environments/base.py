from pydantic import BaseModel, dataclasses, Field, field_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable

from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.skills.base import BaseSkill
from adala.memories.base import ShortTermMemory
from adala.datasets import Dataset, DataFrameDataset


class Environment(BaseModel, ABC):
    """Abstract base class for environments.

    The environment provides a mechanism to obtain ground truth information from raw data and predictions, 
    and also facilitates comparison of ground truth with predictions.

    Attributes:
        Config (class): Configuration for the environment class, allows arbitrary types.
    """
        
    @abstractmethod
    def request_feedback(self, skill: BaseSkill, experience: ShortTermMemory):
        """Request user feedback using predictions and update internal ground truth set."""

    @abstractmethod
    def compare_to_ground_truth(self, skill: BaseSkill, experience: ShortTermMemory) -> ShortTermMemory:
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
        _prediction_column (str): Name of the column containing predictions.

    """
    
    ground_truth_dataset: Union[InternalDataFrame, DataFrameDataset] = Field(default_factory=DataFrameDataset)
    ground_truth_column: str = 'ground_truth'

    _prediction_column: str

    @field_validator('ground_truth_dataset')
    def _validate_ground_truth_dataset(cls, v):
        if isinstance(v, InternalDataFrame):
            return DataFrameDataset(df=v)
        return v

    def request_feedback(self, skill: BaseSkill, experience: ShortTermMemory):
        """In the BasicEnvironment, ground truth is already provided with the input data."""

    def compare_to_ground_truth(self, skill: BaseSkill, experience: ShortTermMemory) -> ShortTermMemory:
        """Compare the predictions with the ground truth using exact matching.

        Args:
            skill (BaseSkill): The skill being evaluated.
            experience (ShortTermMemory): The experience memory containing predictions.

        Returns:
            ShortTermMemory: Updated memory containing evaluation results against ground truth.
        """

        experience = experience.model_copy()

        gt = self.ground_truth_dataset.df[self.ground_truth_column]
        pred = experience.predictions
        # select
        gt = gt[gt.index.isin(pred.index)]
        if gt.empty:
            # return empty memory
            return experience

        gt = gt.to_frame(self.ground_truth_column)

        # compare ground truth with predictions using exact matching
        match_column_name = f'{self.ground_truth_column}__x__{skill.name}'
        evaluations = InternalDataFrameConcat([
            pred,
            (gt[self.ground_truth_column] == pred[skill.name]).rename(match_column_name)
        ], axis=1)
        experience.evaluations = evaluations
        # remember the last column names used in evaluations
        experience.ground_truth_column_name = self.ground_truth_column
        experience.match_column_name = match_column_name
        return experience

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
