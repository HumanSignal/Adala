from pydantic import BaseModel
from typing import List, Optional
from abc import ABC, abstractmethod

from .datasets.base import Dataset, MutableDataset
from .memories.base import ShortTermMemory, LongTermMemory
from .tools.base import Tool


class Skill(BaseModel, ABC):
    instruction: str
    short_term_memory: Optional[ShortTermMemory]
    long_term_memory: Optional[LongTermMemory]
    tools: Optional[List[Tool]]

    @abstractmethod
    def apply(self, dataset: Dataset) -> MutableDataset:
        """
        Apply skill to dataset and return new dataset with skill results (predictions)
        """

    @abstractmethod
    def validate(self, original_dataset: Dataset, predictions: MutableDataset) -> MutableDataset:
        """
        Validate a dataset with predictions and return new Dataset with validation results
        """

    @abstractmethod
    def analyze(self, original_dataset: Dataset, annotated_dataset: MutableDataset) -> None:
        """
        Analyze results and store observations in memory
        """

    @abstractmethod
    def optimize(self) -> None:
        """
        Improve current skill state based on memory and tools
        """

    def evolve(self, dataset: Dataset):
        """
        Apply, validate, analyze and optimize skill.
        """
        agent_predictions = self.apply(dataset)
        annotated_dataset = self.validate(dataset, agent_predictions)
        self.analyze(dataset, annotated_dataset)
        self.optimize()
