from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from .datasets.base import Dataset
from .skills.base import Experience

# following the protocol https://agentprotocol.ai/protocol


class AgentArtifact(BaseModel):
    """
    Base class for agent artifacts
    """
    pass


class AgentStep(BaseModel):
    """
    Base class for agent steps results
    """
    artifacts: List[AgentArtifact]
    is_last: bool


class AgentMemory(BaseModel, ABC):

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


class Agent(BaseModel, ABC):
    """
    Base class for agents.
    """
    dataset: Dataset
    memory: Optional[AgentMemory]

    @abstractmethod
    def step(self, learn=True) -> AgentStep:
        """
        Run agent step and return results
        If learn=False, agent will only act and not learn from environment
        """
