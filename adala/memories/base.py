from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseShortTermMemory(BaseModel, ABC):
    """
    Base class for short-term memories.
    """

    @abstractmethod
    def remember(self, observations: Any):
        """
        Base method for remembering observations in short term memory.
        """

    @abstractmethod
    def retrieve(self) -> Any:
        """
        Base method for retrieving observations from short term memory.
        """


class BaseLongTermMemory(BaseModel, ABC):
    """
    Base class for long-term memories.
    Long-term memories are used to store acquired knowledge and can be shared between agents.
    """

    @abstractmethod
    def remember(self, observations: Any, short_term_memory: Optional[BaseShortTermMemory]):
        """
        Base method for remembering observations in long term memory.
        It is possible to use short term memory to store long term memory observations.
        """

    @abstractmethod
    def retrieve(self, current_observations: Any) -> Any:
        """
        Base method for retrieving observations from long term memory, based on currently observed data.
        """
