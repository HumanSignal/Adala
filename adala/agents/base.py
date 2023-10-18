from pydantic import BaseModel, Field, SkipValidation
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from adala.datasets.base import Dataset, BlankDataset
from adala.runtimes.base import Runtime, LLMRuntime, LLMRuntimeModelType
from adala.memories.base import ShortTermMemory, LongTermMemory

# following the protocol https://agentprotocol.ai/protocol


class AgentStep(BaseModel):
    """
    Base class for agent steps results
    """
    experience: ShortTermMemory
    is_last: bool


class Agent(BaseModel, ABC):
    """
    Base class for agents.
    """
    dataset: Optional[Dataset] = Field(default_factory=lambda: BlankDataset())
    memory: Optional[LongTermMemory] = Field(default=None)
    runtime: Optional[Runtime] = Field(
        default_factory=lambda: LLMRuntime(llm_runtime_type=LLMRuntimeModelType.OpenAI)
    )

    @abstractmethod
    def greet(self) -> str:
        """
        Return agent greeting and description
        """

    @abstractmethod
    def run(self) -> AgentStep:
        """
        Run agent and return results
        """

    @abstractmethod
    def learn(self) -> AgentStep:
        """
        Learn from dataset and return results
        """

