from pydantic import BaseModel, dataclasses
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable

from adala.datasets.base import Dataset
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.memories.base import ShortTermMemory


class Environment(BaseModel, ABC):
    """
    Base class for environments.
    """
    dataset: Dataset
    ground_truth_set: Any

    @abstractmethod
    def request_feedback(self, predictions: InternalDataFrame):
        """
        Using predictions, request feedback from user and update internal ground truth set.
        """

    @abstractmethod
    def compare_to_ground_truth(self, predictions: InternalDataFrame, validation_column: str) -> InternalDataFrame:
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


class DataFrameEnvironment(Environment):
    """
    Environment based on a pandas DataFrame, with a ground truth specified as a column.
    """
    ground_truth_set: InternalDataFrame = None
    ground_truth_column: str = 'ground_truth'
    matching_function: Union[str, Callable] = 'exact_match'

    def request_feedback(self, predictions: InternalDataFrame):
        """
        Using predictions, request feedback from user and save it in `self.ground_truth_set`.
        """
        raise NotImplementedError

    def compare_to_ground_truth(self, predictions: InternalDataFrame, validation_column: str) -> InternalDataFrame:
        """
        Compare predictions with ground truth set and return match results.
        """

        # get ground truth data based on matching index
        gt = self.ground_truth_set[self.ground_truth_set[self.ground_truth_column].notna() & self.ground_truth_set.index.isin(predictions.index)]
        pred = predictions[gt.index]
        pred = pred[pred.notna()]

        if self.matching_function == 'exact_match':
            match = pred[validation_column] == gt[self.ground_truth_column]
        elif isinstance(self.matching_function, Callable):
            match = InternalDataFrameConcat([pred, gt], axis=1).apply(self.matching_function, axis=1)

        evaluations = InternalDataFrameConcat([gt, pred, match.rename('match')], axis=1)
        return evaluations
