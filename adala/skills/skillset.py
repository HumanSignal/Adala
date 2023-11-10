from pydantic import BaseModel, model_validator, field_validator
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional, Mapping
from collections import OrderedDict
from adala.datasets.base import Dataset
from adala.runtimes.base import Runtime
from adala.utils.logs import print_text
from adala.utils.internal_data import InternalDataFrame, InternalSeries, InternalDataFrameConcat, Record
from .base import BaseSkill, LLMSkill
from ._base import Skill


class SkillSet(BaseModel, ABC):
    """
    Represents a collection of interdependent skills aiming to achieve a specific goal.
    
    A skill set breaks down the path to achieve a goal into necessary precursor skills.
    Agents can evolve these skills either in parallel for tasks like self-consistency or 
    sequentially for complex problem decompositions and causal reasoning. In the most generic
    cases, task decomposition can involve a graph-based approach.

    Attributes:
        skills (Dict[str, Skill]): A dictionary of skills in the skill set.
    """
    
    skills: Dict[str, Skill]

    @abstractmethod
    def apply(
        self,
        input: Union[Record, InternalDataFrame],
        runtime: Runtime,
        improved_skill: Optional[str] = None
    ) -> InternalDataFrame:
        """
        Apply the skill set to a dataset using a specified runtime.
        
        Args:
            input (Union[Record, InternalDataFrame]): Input data to apply the skill set to.
            runtime (Runtime): The runtime environment in which to apply the skills.
            improved_skill (Optional[str], optional): Name of the skill to start from (to optimize calculations). Defaults to None.
        Returns:
            InternalDataFrame: Skill predictions.
        """

    def __getitem__(self, skill_name) -> Skill:
        """
        Select skill by name.

        Args:
            skill_name (str): Name of the skill to select.

        Returns:
            BaseSkill: Skill
        """
        return self.skills[skill_name]

    def __setitem__(self, skill_name, skill: Skill):
        """
        Set skill by name.

        Args:
            skill_name (str): Name of the skill to set.
            skill (BaseSkill): Skill to set.
        """
        self.skills[skill_name] = skill

    def get_skill_names(self) -> List[str]:
        """
        Get list of skill names.

        Returns:
            List[str]: List of skill names.
        """
        return list(self.skills.keys())

    def get_skill_outputs(self) -> Dict[str, str]:
        """
        Get dictionary of skill outputs.

        Returns:
            Dict[str, str]: Dictionary of skill outputs. Keys are output names and values are skill names
        """
        return {field: skill.name for skill in self.skills.values() for field in skill.get_output_fields()}


class LinearSkillSet(SkillSet):
    """
    Represents a sequence of skills that are acquired in a specific order to achieve a goal.

    LinearSkillSet ensures that skills are developed in a sequential manner, determined either 
    by the provided skill_sequence or by the lexicographical order of skill names.

    Attributes:
        skills (Union[List[str], Dict[str, str], List[BaseSkill], Dict[str, BaseSkill]]): Provided skills
        skill_sequence (List[str], optional): Ordered list of skill names indicating the order 
                                              in which they should be acquired.
                                              By default, lexographical order of skill names is used.
        input_data_field (Optional[str], optional): Name of the input data field. Defaults to None.

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

    @field_validator('skills', mode='before')
    def skills_validator(cls, v: Union[List[Skill], Dict[str, Skill]]) -> Dict[str, Skill]:
        """
        Validates and converts the skills attribute to a dictionary of skill names to BaseSkill instances.

        Args:
            v (Union[List[str], List[BaseSkill], Dict[str, BaseSkill]]): The skills attribute to validate.

        Returns:
            Dict[str, BaseSkill]: Dictionary mapping skill names to their corresponding BaseSkill instances.
        """
        skills = OrderedDict()
        if not v:
            return skills

        elif isinstance(v, list) and isinstance(v[0], Skill):
            # convert list of skill names to dictionary
            for skill in v:
                skills[skill.name] = skill
        elif isinstance(v, dict):
            skills = v
        else:
            raise ValueError(f"skills must be a list or dictionary, not {type(skills)}")
        return skills

    @model_validator(mode='after')
    def skill_sequence_validator(self) -> 'LinearSkillSet':
        """
        Validates and sets the default order for the skill sequence if not provided.
        
        Returns:
            LinearSkillSet: The current instance with updated skill_sequence attribute.
        """
        if self.skill_sequence is None:
            # use default skill sequence defined by lexicographical order
            self.skill_sequence = list(self.skills.keys())
        if len(self.skill_sequence) != len(self.skills):
            raise ValueError(f"skill_sequence must contain all skill names - "
                             f"length of skill_sequence is {len(self.skill_sequence)} "
                             f"while length of skills is {len(self.skills)}")
        return self

    def apply(
        self,
        input: Union[Record, InternalDataFrame],
        runtime: Runtime,
        improved_skill: Optional[str] = None,
    ) -> InternalDataFrame:
        """
        Sequentially applies each skill on the dataset, enhancing the agent's experience.
        
        Args:
            input (InternalDataFrame): Input dataset.
            runtime (Runtime): The runtime environment in which to apply the skills.
            improved_skill (Optional[str], optional): Name of the skill to improve. Defaults to None.
        Returns:
            InternalDataFrame: Skill predictions.
        """
        if improved_skill:
            # start from the specified skill, assuming previous skills have already been applied
            skill_sequence = self.skill_sequence[self.skill_sequence.index(improved_skill):]
        else:
            skill_sequence = self.skill_sequence
        skill_input = input
        for i, skill_name in enumerate(skill_sequence):
            skill = self.skills[skill_name]
            # use input dataset for the first node in the pipeline
            print_text(f"Applying skill: {skill_name}")
            skill_output = skill.apply(skill_input, runtime)
            if isinstance(skill_output, InternalDataFrame) and isinstance(skill_input, InternalDataFrame):
                # Columns to drop from skill_input because they are also in skill_output
                cols_to_drop = set(skill_output.columns) & set(skill_input.columns)
                skill_input_reduced = skill_input.drop(columns=cols_to_drop)

                skill_input = skill_output.merge(
                    skill_input_reduced,
                    left_index=True,
                    right_index=True,
                    how='inner'
                )
            elif isinstance(skill_output, InternalDataFrame) and isinstance(skill_input, dict):
                skill_input = skill_output
            elif isinstance(skill_output, dict) and isinstance(skill_input, InternalDataFrame):
                skill_input = skill_output
            elif isinstance(skill_output, dict) and isinstance(skill_input, dict):
                skill_input = dict(skill_output, **skill_input)
            else:
                raise ValueError(f"Unsupported input type: {type(skill_input)} and output type: {type(skill_output)}")

        return skill_input

    def select_skill_to_improve(
        self,
        accuracy: Mapping,
        accuracy_threshold: Optional[float] = 1.0
    ) -> Optional[Skill]:
        """
        Selects the skill with the lowest accuracy to improve.

        Args:
            accuracy (Mapping): Accuracy of each skill.
            accuracy_threshold (Optional[float], optional): Accuracy threshold. Defaults to 1.0.
        Returns:
            Optional[BaseSkill]: Skill to improve. None if no skill to improve.
        """

        for skill_output, skill_name in self.get_skill_outputs():
            if accuracy[skill_output] < accuracy_threshold:
                return self.skills[skill_name]

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

    In a ParallelSkillSet, each skill can be developed independently of the others. This is useful
    for agents that require multiple, diverse capabilities, or tasks where each skill contributes a piece of
    the overall solution.

    Examples: 
        Create a ParallelSkillSet with a list of skills specified as BaseSkill instances
        >>> from adala.skills import ParallelSkillSet, TextClassificationSkill, TextGenerationSkill
        >>> skillset = ParallelSkillSet(skills=[TextClassificationSkill(name='Classify sentiment', instructions='Classify the sentiment'), TextGenerationSkill(name='Summarize text', instructions='Generate a summar')])

        Create a ParallelSkillSet with a dictionary of skill names to BaseSkill instances
        >>> from adala.skills import ParallelSkillSet, TextClassificationSkill, TextGenerationSkill
        >>> skillset = ParallelSkillSet(skills={'sentiment_analysis': TextClassificationSkill(name='Classify sentiment', instructions='Classify the sentiment'),'text_summary': TextGenerationSkill(name='Summarize text', instructions='Generate a summary')})
    """

    @field_validator("skills", mode="before")
    @classmethod
    def skills_validator(
        cls, v: Union[List[BaseSkill], Dict[str, BaseSkill]]
    ) -> Dict[str, BaseSkill]:
        """
        Validates and converts the skills attribute to a dictionary of skill names to BaseSkill instances.

        Args:
            v (List[BaseSkill], Dict[str, BaseSkill]]): The skills attribute to validate.

        Returns:
            Dict[str, BaseSkill]: Dictionary mapping skill names to their corresponding BaseSkill instances.
        """
        skills = OrderedDict()
        if not v:
            return skills

        if isinstance(v, list) and isinstance(v[0], BaseSkill):
            # convert list of skill names to dictionary
            for skill in v:
                skills[skill.name] = skill
        elif isinstance(v, dict):
            skills = v
        else:
            raise ValidationError(
                f"skills must be a list or dictionary, not {type(skills)}"
            )
        return skills

    def apply(
        self,
        dataset: Union[Dataset, InternalDataFrame],
        runtime: Runtime,
        improved_skill: Optional[str] = None,
    ) -> InternalDataFrame:
        """
        Applies each skill on the dataset, enhancing the agent's experience.

        Args:
            dataset (Dataset): The dataset to apply the skills on.
            runtime (Runtime): The runtime environment in which to apply the skills.
            improved_skill (Optional[str], optional): Unused in ParallelSkillSet. Defaults to None.
        Returns:
            InternalDataFrame: Skill predictions.
        """
        predictions = None

        for i, skill_name in enumerate(self.skills.keys()):
            skill = self.skills[skill_name]
            # use input dataset for the first node in the pipeline
            input_dataset = dataset if i == 0 else predictions
            print_text(f"Applying skill: {skill_name}")
            predictions = skill.apply(input_dataset, runtime)

        return predictions

    def select_skill_to_improve(
        self, accuracy: Mapping, accuracy_threshold: Optional[float] = 0.9
    ) -> Optional[BaseSkill]:
        """
        Selects the skill with the lowest accuracy to improve.

        Args:
            accuracy (Mapping): Accuracy of each skill.
            accuracy_threshold (Optional[float], optional): Accuracy threshold. Defaults to 1.0.
        Returns:
            Optional[BaseSkill]: Skill to improve. None if no skill to improve.
        """
        skills_below_threshold = [
            skill_name
            for skill_name in self.skills.keys()
            if accuracy[skill_name] < accuracy_threshold
        ]
        if skills_below_threshold:
            weakest_skill_name = min(skills_below_threshold, key=accuracy.get)
            return self.skills[weakest_skill_name]
