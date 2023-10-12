from pydantic import BaseModel
from typing import List, Optional
from abc import ABC, abstractmethod

from .datasets.base import Dataset, MutableDataset
from .tools.base import Tool


class Experience(BaseModel):
    """
    Base class for skill experiences - results of evolving skills based on a dataset
    """


class Skill(BaseModel, ABC):
    name: str
    instruction: str
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
    def remember(self, annotated_dataset: MutableDataset) -> None:
        """
        Remember observations from validation results in short term memory
        """

    @abstractmethod
    def analyze(self, original_dataset: Dataset, annotated_dataset: MutableDataset) -> Experience:
        """
        Analyze results and return observed experience - it will be stored in long term memory
        """

    @abstractmethod
    def optimize(self) -> None:
        """
        Improve current skill state based on memory and tools
        """

    def evolve(self, dataset: Dataset) -> Experience:
        """
        Apply, validate, analyze and optimize skill.
        """
        agent_predictions = self.apply(dataset)
        annotated_dataset = self.validate(dataset, agent_predictions)
        self.remember(annotated_dataset)
        experience = self.analyze(dataset, annotated_dataset)
        self.optimize()
        return experience
