from pydantic import BaseModel
from typing import List, Optional, Any
from abc import ABC, abstractmethod

from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import Memory, Experience


class Skill(BaseModel, ABC):
    name: str
    instructions: str
    description: Optional[str]

    _previous_instructions: Optional[List[str]] = []

    @abstractmethod
    def apply(self, dataset: Dataset, runtime: Runtime) -> Dataset:
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
        self, original_dataset: Dataset, evaluation: Dataset, memory: Memory
    ) -> Experience:
        """
        Analyze results and return observed experience - it will be stored in long term memory
        """

    @abstractmethod
    def improve(self, dataset: Dataset, experience: Experience) -> None:
        """
        Improve current skill state based on current experience
        """

    def learn(self, dataset: Dataset, runtime: Runtime, memory: Memory) -> Experience:
        """
        Apply, validate, analyze and optimize skill.
        """
        predictions = self.apply(dataset, runtime=runtime)
        annotated_dataset = self.evaluate(dataset, predictions)
        experience = self.analyze(dataset, annotated_dataset, memory)
        self.improve(dataset, experience)
        return experience
