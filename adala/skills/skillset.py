from pydantic import BaseModel, model_validator
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import ShortTermMemory
from .base import BaseSkill


class SkillSet(BaseModel, ABC):
    """
    Represents a collection of interdependent skills aiming to achieve a specific goal.
    
    A skill set breaks down the path to achieve a goal into necessary precursor skills.
    Agents can evolve these skills either in parallel for tasks like self-consistency or 
    sequentially for complex problem decompositions and causal reasoning. In the most generic
    cases, task decomposition can involve a graph-based approach.

    Args:
        skills (Dict[str, BaseSkill]): Dictionary mapping skill names to their corresponding 
                                       BaseSkill instances.
    """
    
    skills: Dict[str, BaseSkill]

    @abstractmethod
    def apply(self, dataset: Dataset, runtime: Runtime, experience: Optional[ShortTermMemory] = None) -> ShortTermMemory:
        """
        Apply the skill set to a dataset using a specified runtime.
        
        Args:
            dataset (Dataset): The dataset to apply the skill set to.
            runtime (Runtime): The runtime environment in which to apply the skills.
            experience (Optional[ShortTermMemory], optional): Existing experience data. Defaults to None.
            
        Returns:
            ShortTermMemory: Updated experience after applying the skill set.
        """

    @abstractmethod
    def select_skill_to_improve(self, experience: ShortTermMemory) -> BaseSkill:
        """
        Select the next skill to enhance based on the current experience.
        
        Args:
            experience (ShortTermMemory): Current experience data.
            
        Returns:
            BaseSkill: Skill selected for improvement.
        """


class LinearSkillSet(SkillSet):
    """
    Represents a sequence of skills that are acquired in a specific order to achieve a goal.

    LinearSkillSet ensures that skills are developed in a sequential manner, determined either 
    by the provided skill_sequence or by the lexicographical order of skill names.

    Args:
        skill_sequence (List[str], optional): Ordered list of skill names indicating the order 
                                              in which they should be acquired.
    """
    
    skill_sequence: List[str] = None

    @model_validator(mode='after')
    def skill_sequence_validator(self):
        """
        Validates and sets the default order for the skill sequence if not provided.
        
        Returns:
            LinearSkillSet: The current instance with updated skill_sequence attribute.
        """
        
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
        Sequentially applies each skill on the dataset, enhancing the agent's experience.
        
        Args:
            dataset (Dataset): The dataset to apply the skills on.
            runtime (Runtime): The runtime environment in which to apply the skills.
            experience (Optional[ShortTermMemory], optional): Existing experience data. Defaults to None.
            
        Returns:
            ShortTermMemory: Updated experience after sequentially applying the skills.
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
        Picks the next skill for improvement in the sequence.
        
        Args:
            experience (ShortTermMemory): Current experience data.
            
        Returns:
            BaseSkill: The next skill selected for improvement.
        """
        
        # TODO: implement real logic for skill selection
        return self.skills[self.skill_sequence[0]]


class ParallelSkillSet(SkillSet):
    """
    Represents a set of skills that are acquired simultaneously to reach a goal.
    """
    
    pass
