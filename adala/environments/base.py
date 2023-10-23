from pydantic import BaseModel, dataclasses, Field
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable

from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.skills.base import BaseSkill
from adala.memories.base import ShortTermMemory
from adala.datasets import Dataset, DataFrameDataset


class Environment(BaseModel, ABC):
    """
    Base class for environments.
    Each environment differs by the way it obtains ground truth information from raw data and predictions,
    and the way it compares ground truth with predictions.
    Environment uses predictions index as a way to match predictions with ground truth.
    """

    @abstractmethod
    def request_feedback(self, skill: BaseSkill, experience: ShortTermMemory):
        """
        Using predictions, request feedback from user and update internal ground truth set.
        """

    @abstractmethod
    def compare_to_ground_truth(self, skill: BaseSkill, experience: ShortTermMemory) -> ShortTermMemory:
        """
        Compare predictions with ground truth set and return match results.
        """

    @abstractmethod
    def as_dataset(self) -> Dataset:
        """
        Return environment as a dataset.
        """

    @abstractmethod
    def save(self):
        """
        Save environment state.
        """

    @abstractmethod
    def restore(self):
        """
        Restore environment state.
        """

    class Config:
        arbitrary_types_allowed = True


class BasicEnvironment(Environment):
    """
    BasicEnvironment assumes ground truth signal comes explicitly with the input data.
    Once new ground truth points are added, they are saved in `self.ground_truth_set`.
    To compare with ground truth, exact matching is used.
    """
    ground_truth_dataset: DataFrameDataset = Field(default_factory=DataFrameDataset)
    ground_truth_column: str = 'ground_truth'

    _prediction_column: str

    def request_feedback(self, skill: BaseSkill, experience: ShortTermMemory):
        """
        For BasicEnvironment, ground truth is already provided with the input data.
        """

    def compare_to_ground_truth(self, skill: BaseSkill, experience: ShortTermMemory) -> ShortTermMemory:
        """
        Compare predictions with ground truth set and return match results.
        """

        experience = experience.model_copy()

        gt = self.ground_truth_dataset.df[self.ground_truth_column]
        pred = experience.predictions
        # select
        gt = gt[gt.index.isin(pred.index)]
        if gt.empty:
            # return empty memory
            return ShortTermMemory()

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
        return self.ground_truth_dataset

    def save(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError
