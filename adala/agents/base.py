from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from .datasets.base import Dataset

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


class Agent(BaseModel, ABC):
    """
    Base class for agents.
    """
    dataset: Dataset

    @abstractmethod
    def step(self, learn=True) -> AgentStep:
        """
        Run agent step and return results
        If learn=False, agent will only act and not learn from environment
        """
