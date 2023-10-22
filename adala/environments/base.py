from pydantic import BaseModel
from abc import ABC, abstractmethod

from adala.datasets.base import Dataset
from adala.utils.internal_data import InternalDataFrame
from adala.memories.base import ShortTermMemory


class Environment(BaseModel, ABC):
    """
    Base class for environments.
    """
    dataset: Dataset

    def request_feedback(self, experience: ShortTermMemory) -> InternalDataFrame:
        """
        Return feedback from user on predictions.
        """
        # TODO: this is a stub function - more generic implementation is needed
        return self.dataset.get_ground_truth(experience.predictions)
