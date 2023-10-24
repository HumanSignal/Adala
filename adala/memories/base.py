from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel
from adala.datasets.base import Dataset, InternalDataFrame
from rich import print

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

    def reset(self):
        self.predictions = None
        self.evaluations = None
        self.errors = None
        self.accuracy = None
        self.initial_instructions = None
        self.updated_instructions = None

    def __rich__(self):
        text = '[bold blue]Agent Experience:[/bold blue]\n\n'
        if self.predictions is not None:
            text += f'\n[bold]Predictions[/bold]\n{self.predictions}'
        if self.evaluations is not None:
            text += f'\n[bold]Evaluations[/bold]\n{self.evaluations}'
        if self.errors is not None:
            text += f'\n[bold]Errors[/bold]\n{self.errors}'
        if self.accuracy is not None:
            text += f'\n[bold]Accuracy[/bold]\n{self.accuracy}'
        if self.initial_instructions is not None:
            text += f'\n[bold]Initial Instructions[/bold]\n{self.initial_instructions}'
        if self.updated_instructions is not None:
            text += f'\n[bold]Updated Instructions[/bold]\n{self.updated_instructions}'
        return text

    def display(self):
        print(self)


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
