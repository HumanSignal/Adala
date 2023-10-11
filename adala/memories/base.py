from pydantic import BaseModel
from .analyzers.base import Observations


class ShortTermMemory(BaseModel):
    """
    Base class for short-term memories.
    """

    def remember(self, observations: Observations):
        pass


class LongTermMemory(BaseModel):
    """
    Base class for long-term memories.
    Long-term memories are used to store acquired knowledge and can be shared between agents.
    """
    def remember(self, observations: Observations, short_term_memory: ShortTermMemory):
        pass
