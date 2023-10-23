from pydantic import BaseModel, dataclasses, Field
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable

from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.skills.base import BaseSkill
from adala.memories.base import ShortTermMemory
from adala.datasets.dataframe import DataFrameDataset


class Environment(BaseModel, ABC):
    """
    Base class for environments.
    Each environment differs by the way it obtains ground truth information from raw data and predictions,
    and the way it compares ground truth with predictions.
    Environment uses predictions index as a way to match predictions with ground truth.
    """
    ground_truth_set: Any

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
        To extract ground truth from predictions, we simply take the ground truth column,
        and add extracted ground truth to ground truth set.
        """

    def compare_to_ground_truth(self, skill: BaseSkill, experience: ShortTermMemory) -> ShortTermMemory:
        """
        Compare predictions with ground truth set and return match results.
        """
        gt = self.ground_truth_dataset.df[self.ground_truth_column]
        pred = experience.predictions[self._prediction_column]
        # select
        gt = gt[gt.index.isin(pred.index)]
        if not gt.empty:
            gt = gt.to_frame(self.ground_truth_column)
            if self.ground_truth_set.empty:
                self.ground_truth_set = gt
            else:
                # TODO: control the size of ground truth set to avoid memory issues
                self.ground_truth_set = InternalDataFrameConcat([self.ground_truth_set, gt], axis=0)

        experience = experience.model_copy()
        predictions = experience.predictions

        # TODO: support multiple prediction columns
        prediction_column = skill.validation_fields[0]

        # get ground truth data based on matching index
        gt = self.ground_truth_set[self.ground_truth_set.index.isin(experience.predictions.index)]
        pred = predictions[predictions.index.isin(gt.index)]

        # compare ground truth with predictions using exact matching
        match_column_name = f'{self.ground_truth_column}__x__{prediction_column}'
        evaluations = InternalDataFrameConcat([
            pred,
            (gt[self.ground_truth_column] == pred[prediction_column]).rename(match_column_name)
        ], axis=1)
        experience.evaluations = evaluations
        # remember the last column names used in evaluations
        experience.prediction_column_name = prediction_column
        experience.ground_truth_column_name = self.ground_truth_column
        experience.match_column_name = match_column_name
        return experience

    def save(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError
