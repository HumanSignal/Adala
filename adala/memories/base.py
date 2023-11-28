from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING, Dict, List
from pydantic import BaseModel, Field
from adala.utils.internal_data import InternalDataFrame


class Memory(BaseModel, ABC):

    """
    Base class for memories.
    """

    @abstractmethod
    def remember(self, observation: str, data: Dict):
        """
        Base method for remembering experiences in long term memory.
        """

    def remember_many(self, observations: List[str], data: List[Dict]):
        """
        Base method for remembering experiences in long term memory.
        """
        for observation, d in zip(observations, data):
            self.remember(observation, d)

    @abstractmethod
    def retrieve(self, observation: str) -> Any:
        """
        Base method for retrieving past experiences from long term memory, based on current observations
        """

    def retrieve_many(self, observations: List[str]) -> List[Any]:
        """
        Base method for retrieving past experiences from long term memory, based on current observations
        """
        return [self.retrieve(observation) for observation in observations]

    @abstractmethod
    def clear(self):
        """
        Base method for clearing memory.
        """