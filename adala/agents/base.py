from pydantic import BaseModel
from typing import List

from .skills.base import Skill
from .datasets.base import Dataset, MutableDataset
from .optimizers.base import Optimizer
from .validators.base import Validator
from .analyzers.base import Analyzer
from .memories.base import ShortTermMemory, LongTermMemory
from .tools.base import Tool


class Agent(BaseModel):
    skill: Skill
    validator: Validator
    analyzer: Analyzer
    optimizer: Optimizer
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory
    tools: List[Tool] = []

    _skill_history: List[Skill] = []

    def act(self, dataset: Dataset) -> MutableDataset:
        return self.skill.apply(dataset)

    def reflect(self, original_dataset: Dataset, predictions: MutableDataset):
        ground_truth = self.validator.validate(original_dataset, predictions)
        current_observations = self.analyzer.analyze(ground_truth)
        self.short_term_memory.remember(current_observations)
        self.long_term_memory.remember(current_observations, self.short_term_memory)
        optimized_skill = self.optimizer.optimize(
            self.skill,
            self.short_term_memory,
            self.long_term_memory,
            self.tools
        )
        self._skill_history.append(self.skill)
        self.skill = optimized_skill
