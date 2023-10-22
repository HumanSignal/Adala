from pydantic import BaseModel, model_validator
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import ShortTermMemory
from .base import BaseSkill


class SkillSet(BaseModel, ABC):
    """
    Collection of interdependent skills
    Skill set aims at achieving a particular goal,
    therefore it splits the goal path into the skills that are needed to be acquired first.
    Agents can evolve skills in parallel (e.g. for self-consistency)
     or in sequence (e.g. for complex problem decomposition and causal reasoning)
    Most generic task can involve graph decomposition
    """
    skills: Dict[str, BaseSkill]

    @abstractmethod
    def apply(self, dataset: Dataset, runtime: Runtime, experience: Optional[ShortTermMemory] = None) -> ShortTermMemory:
        """
        Apply skill set to a dataset using a runtime
        """

    @abstractmethod
    def select_skill_to_improve(self, experience: ShortTermMemory) -> BaseSkill:
        """
        Select next skill to improve based on current experience
        """


class LinearSkillSet(SkillSet):
    """
    Linear skill set is a sequence of skills that need to be acquired in order to achieve a goal
    """
    skill_sequence: List[str] = None

    @model_validator(mode='after')
    def skill_sequence_validator(self):
        if self.skill_sequence is None:
            # use default skill sequence defined by lexicographical order
            self.skill_sequence = sorted(self.skills.keys())
        return self

    def apply(
        self, dataset: Dataset,
        runtime: Runtime,
        experience: Optional[ShortTermMemory] = None
    ) -> ShortTermMemory:
        """
        Apply skill set to a dataset using a runtime
        """
        if experience is None:
            experience = ShortTermMemory(dataset=dataset)
        else:
            experience = experience.model_copy()

        for skill_name in self.skill_sequence:
            skill = self.skills[skill_name]
            experience = skill.apply(dataset, runtime, experience)
        return experience

    def select_skill_to_improve(self, experience: ShortTermMemory) -> BaseSkill:
        """
        Select next skill to improve based on current experience
        """
        # TODO: implement real logic for skill selection
        return self.skills[self.skill_sequence[0]]


class ParallelSkillSet(SkillSet):
    """
    Parallel skill set is a set of skills that can be acquired in parallel to achieve a goal
    """
    pass
