from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel
from adala.datasets.base import Dataset, InternalDataFrame

if TYPE_CHECKING:
    from adala.skills.skillset import SkillSet


class ShortTermMemory(BaseModel):
    """
    Base class for short term memory storage
    """
    dataset: Dataset = None
    predictions: InternalDataFrame = None
    evaluations: InternalDataFrame = None
    ground_truth_column_name: str = None
    match_column_name: str = None
    errors: InternalDataFrame = None
    accuracy: float = None
    initial_instructions: str = None
    updated_instructions: str = None

    class Config:
        arbitrary_types_allowed = True


class LongTermMemory(BaseModel, ABC):

    """
    Base class for long-term memories.
    Long-term memories are used to store acquired knowledge and can be shared between agents.
    """

    @abstractmethod
    def remember(self, experience: ShortTermMemory, skills: SkillSet):
        """
        Base method for remembering experiences in long term memory.
        """

    @abstractmethod
    def retrieve(self, observations: ShortTermMemory) -> ShortTermMemory:
        """
        Base method for retrieving past experiences from long term memory, based on current observations
        """
