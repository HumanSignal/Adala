from pydantic import BaseModel
from typing import List, Optional, Any
from abc import ABC, abstractmethod

from adala.datasets.base import Dataset


class Experience(BaseModel):
    """
    Base class for skill experiences - results of evolving skills based on a dataset
    """


class LongTermMemory(BaseModel, ABC):

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


class Skill(BaseModel, ABC):
    name: str
    instructions: str
    description: Optional[str]

    _previous_instructions: Optional[List[str]]

    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        """
        Apply skill to dataset and return new dataset with skill results (predictions)
        """

    @abstractmethod
    def evaluate(self, original_dataset: Dataset, predictions: Dataset) -> Dataset:
        """
        Test predictions and return new Dataset with evaluation results - e.g. error examples
        """

    @abstractmethod
    def analyze(
        self, original_dataset: Dataset, evaluation: Dataset, memory: LongTermMemory
    ) -> Experience:
        """
        Analyze results and return observed experience - it will be stored in long term memory
        """

    @abstractmethod
    def improve(self, experience: Experience) -> None:
        """
        Improve current skill state based on current experience
        """

    def learn(self, dataset: Dataset, long_term_memory: LongTermMemory) -> Experience:
        """
        Apply, validate, analyze and optimize skill.
        """
        predictions = self.apply(dataset)
        annotated_dataset = self.evaluate(dataset, predictions)
        experience = self.analyze(dataset, annotated_dataset, long_term_memory)
        self.improve(dataset, experience)
        return experience
