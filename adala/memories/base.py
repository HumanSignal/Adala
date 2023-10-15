from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel


class Experience(BaseModel):
    """
    Base class for skill experiences - results of evolving skills based on a dataset
    """


class Memory(BaseModel, ABC):

    """
    Base class for long-term memories.
    Long-term memories are used to store acquired knowledge and can be shared between agents.
    """

    @abstractmethod
    def remember(self, experience: Experience):
        """
        Base method for remembering experiences in long term memory.
        """

    @abstractmethod
    def retrieve(self, observations: Any) -> Experience:
        """
        Base method for retrieving past experiences from long term memory, based on current observations
        """
