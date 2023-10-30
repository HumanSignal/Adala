from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING, Dict

from pydantic import BaseModel, Field
from adala.datasets.base import Dataset, InternalDataFrame
from rich import print


class Memory(BaseModel, ABC):

    """
    Base class for long-term memories.
    Long-term memories are used to store acquired knowledge and can be shared between agents.
    """

    @abstractmethod
    def remember(self, observation: str, experience: Any):
        """
        Base method for remembering experiences in long term memory.
        """

    @abstractmethod
    def retrieve(self, observation: str) -> Any:
        """
        Base method for retrieving past experiences from long term memory, based on current observations
        """
