from pydantic import BaseModel, model_validator, field_validator
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.memories.base import ShortTermMemory
from .base import BaseSkill, LLMSkill


class SkillSet(BaseModel, ABC):
    """
    Represents a collection of interdependent skills aiming to achieve a specific goal.
    
    A skill set breaks down the path to achieve a goal into necessary precursor skills.
    Agents can evolve these skills either in parallel for tasks like self-consistency or 
    sequentially for complex problem decompositions and causal reasoning. In the most generic
    cases, task decomposition can involve a graph-based approach.

    Args:
        skills (Union[List[str], Dict[str, str], List[BaseSkill], Dict[str, BaseSkill]]): Provided skills
    """
    
    skills: Union[List[str], Dict[str, str], List[BaseSkill], Dict[str, BaseSkill]]

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
        skills (Union[List[str], Dict[str, str], List[BaseSkill], Dict[str, BaseSkill]]): Provided skills
        skill_sequence (List[str], optional): Ordered list of skill names indicating the order 
                                              in which they should be acquired.
                                              By default, lexographical order of skill names is used.

    Examples:
        Create a LinearSkillSet with a list of skills specified as strings:
        >>> from adala.skills import LinearSkillSet
        >>> skillset = LinearSkillSet(skills=['Extract keywords', 'Classify keywords', 'Create structured output'])

        Create a LinearSkillSet with a list of skills specified as BaseSkill instances:
        >>> from adala.skills import LinearSkillSet, TextGenerationSkill
        >>> skillset = LinearSkillSet(skills=[TextGenerationSkill(name='Generate text', instructions='Generate text from keywords'),])

        Create a LinearSkillSet with a dictionary of skill names to instructions:
        >>> from adala.skills import LinearSkillSet
        >>> skillset = LinearSkillSet(skills={'extract': 'Extract keywords from text', 'classify': 'Classify keywords', 'structured_output': 'Create structured output from keywords'})

    """
    
    skill_sequence: List[str] = None

    @field_validator('skills')
    def skills_validator(cls, v: Union[List[str], List[BaseSkill], Dict[str, BaseSkill]]) -> Dict[str, BaseSkill]:
        """
        Validates and converts the skills attribute to a dictionary of skill names to BaseSkill instances.

        Args:
            skills (Union[List[str], List[BaseSkill], Dict[str, BaseSkill]]): The skills attribute to validate.

        Returns:
            Dict[str, BaseSkill]: Dictionary mapping skill names to their corresponding BaseSkill instances.
        """
        if not v:
            return {}
        skills = {}
        if isinstance(v, list) and isinstance(v[0], str):
            # if list of strings presented, they are interpreted as skill instructions
            input_data_field = 'text'
            for i, instructions in enumerate(v):
                skill_name = f"skill_{i}"
                skills[skill_name] = LLMSkill(
                    name=skill_name,
                    instructions=instructions,
                    input_data_field=input_data_field
                )
                # Linear skillset creates skills pipeline - update input_data_field for next skill
                input_data_field = skill_name
        elif isinstance(v, dict) and isinstance(v[list(v.keys())[0]], str):
            # if dictionary of strings presented, they are interpreted as skill instructions
            input_data_field = 'text'
            for skill_name, instructions in v.items():
                skills[skill_name] = LLMSkill(
                    name=skill_name,
                    instructions=instructions,
                    input_data_field=input_data_field
                )
                # Linear skillset creates skills pipeline - update input_data_field for next skill
                input_data_field = skill_name
        elif isinstance(v, list) and isinstance(v[0], BaseSkill):
            # convert list of skill names to dictionary
            skills = {skill.name: skill for skill in v}
        elif isinstance(v, dict):
            skills = v
        else:
            raise ValueError(f"skills must be a list or dictionary, not {type(skills)}")
        return skills

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

        for i, skill_name in enumerate(self.skill_sequence):
            skill = self.skills[skill_name]
            # use input dataset for the first node in the pipeline
            input_dataset = dataset if i == 0 else experience.predictions
            experience = skill.apply(input_dataset, runtime, experience)
        
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
        return self.skills[self.skill_sequence[-1]]

    def __rich__(self):
        """Returns a rich representation of the skill."""
        # TODO: move it to a base class and use repr derived from Skills
        text = f"[bold blue]Total Agent Skills: {len(self.skills)}[/bold blue]\n\n"
        for skill in self.skills.values():
            text += f'[bold underline green]{skill.name}[/bold underline green]\n' \
                    f'[green]{skill.instructions}[green]\n'
        return text


class ParallelSkillSet(SkillSet):
    """
    Represents a set of skills that are acquired simultaneously to reach a goal.
    """
    
    pass
