from pydantic import BaseModel, model_validator, field_validator
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional, Mapping, Type
from collections import OrderedDict
from adala.runtimes.base import Runtime, AsyncRuntime
from adala.utils.logs import print_text, print_dataframe
from adala.utils.internal_data import (
    InternalDataFrame,
    InternalSeries,
    InternalDataFrameConcat,
    Record,
)
from ._base import (
    Skill,
    TransformSkill,
    SampleTransformSkill,
    AnalysisSkill,
    SynthesisSkill
)


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

    @field_validator("skills", mode="before")
    def skills_validator(cls, v: Union[List, Dict]) -> Dict[str, Skill]:
        """
        Validates and converts the skills attribute to a dictionary of skill names to BaseSkill instances.

        Args:
            v (Union[List[Skill], Dict[str, Skill]]): The skills attribute to validate and convert.

        Returns:
            Dict[str, BaseSkill]: Dictionary mapping skill names to their corresponding BaseSkill instances.
        """
        skills = OrderedDict()
        if not v:
            return skills

        elif isinstance(v, list):
            if isinstance(v[0], Skill):
                # convert list of skill names to dictionary
                for skill in v:
                    skills[skill.name] = skill
            elif isinstance(v[0], dict):
                # convert list of skill dictionaries to dictionary
                for skill in v:
                    if 'type' not in skill:
                        raise ValueError("Skill dictionary must contain a 'type' key")
                    skills[skill["name"]] = Skill.create_from_registry(skill.pop('type'), **skill)
        elif isinstance(v, dict):
            skills = v
        else:
            raise ValueError(f"skills must be a list or dictionary, not {type(skills)}")
        return skills

    @abstractmethod
    def apply(
        self,
        input: Union[Record, InternalDataFrame],
        runtime: Runtime,
        improved_skill: Optional[str] = None,
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
        return {
            field: skill.name
            for skill in self.skills.values()
            for field in skill.get_output_fields()
        }


class LinearSkillSet(SkillSet):
    """
    Represents a sequence of skills that are acquired in a specific order to achieve a goal.

    LinearSkillSet ensures that skills are applied in a sequential manner.

    Attributes:
        skills (Union[List[Skill], Dict[str, Skill]]): Provided skills
        skill_sequence (List[str], optional): Ordered list of skill names indicating the order
                                              in which they should be acquired.

    Examples:

        Create a LinearSkillSet with a list of skills specified as BaseSkill instances:
        >>> from adala.skills import LinearSkillSet, TransformSkill, AnalysisSkill, ClassificationSkill
        >>> skillset = LinearSkillSet(skills=[TransformSkill(), ClassificationSkill(), AnalysisSkill()])
    """

    skill_sequence: List[str] = None

    @model_validator(mode="after")
    def skill_sequence_validator(self) -> "LinearSkillSet":
        """
        Validates and sets the default order for the skill sequence if not provided.

        Returns:
            LinearSkillSet: The current instance with updated skill_sequence attribute.
        """
        if self.skill_sequence is None:
            # use default skill sequence defined by lexicographical order
            self.skill_sequence = list(self.skills.keys())
        if len(self.skill_sequence) != len(self.skills):
            raise ValueError(
                f"skill_sequence must contain all skill names - "
                f"length of skill_sequence is {len(self.skill_sequence)} "
                f"while length of skills is {len(self.skills)}"
            )
        return self

    def apply(
        self,
        input: Union[Record, InternalDataFrame],
        runtime: Runtime,
        improved_skill: Optional[str] = None,
    ) -> InternalDataFrame:
        """
        Sequentially applies each skill on the dataset.

        Args:
            input (InternalDataFrame): Input dataset.
            runtime (Runtime): The runtime environment in which to apply the skills.
            improved_skill (Optional[str], optional): Name of the skill to improve. Defaults to None.
        Returns:
            InternalDataFrame: Skill predictions.
        """
        if improved_skill:
            # start from the specified skill, assuming previous skills have already been applied
            skill_sequence = self.skill_sequence[
                self.skill_sequence.index(improved_skill) :
            ]
        else:
            skill_sequence = self.skill_sequence
        skill_input = input
        for i, skill_name in enumerate(skill_sequence):
            skill = self.skills[skill_name]
            # use input dataset for the first node in the pipeline
            print_text(f"Applying skill: {skill_name}")
            skill_output = skill.apply(skill_input, runtime)
            print_dataframe(skill_output)
            if isinstance(skill, TransformSkill):
                # Columns to drop from skill_input because they are also in skill_output
                cols_to_drop = set(skill_output.columns) & set(skill_input.columns)
                skill_input_reduced = skill_input.drop(columns=cols_to_drop)

                skill_input = skill_input_reduced.merge(
                    skill_output, left_index=True, right_index=True, how="inner"
                )
            elif isinstance(skill, (AnalysisSkill, SynthesisSkill)):
                skill_input = skill_output
            else:
                raise ValueError(f"Unsupported skill type: {type(skill)}")
        if isinstance(skill_input, InternalSeries):
            skill_input = skill_input.to_frame().T
        return skill_input

    async def aapply(
        self,
        input: Union[Record, InternalDataFrame],
        runtime: AsyncRuntime,
        improved_skill: Optional[str] = None,
    ) -> InternalDataFrame:
        """
        Sequentially and asynchronously applies each skill on the dataset.

        Args:
            input (InternalDataFrame): Input dataset.
            runtime (AsyncRuntime): The runtime environment in which to apply the skills.
            improved_skill (Optional[str], optional): Name of the skill to improve. Defaults to None.
        Returns:
            InternalDataFrame: Skill predictions.
        """
        if improved_skill:
            # start from the specified skill, assuming previous skills have already been applied
            skill_sequence = self.skill_sequence[
                self.skill_sequence.index(improved_skill) :
            ]
        else:
            skill_sequence = self.skill_sequence
        skill_input = input
        for i, skill_name in enumerate(skill_sequence):
            skill = self.skills[skill_name]
            # use input dataset for the first node in the pipeline
            print_text(f"Applying skill: {skill_name}")
            skill_output = await skill.aapply(skill_input, runtime)
            print_dataframe(skill_output)
            if isinstance(skill, TransformSkill):
                # Columns to drop from skill_input because they are also in skill_output
                cols_to_drop = set(skill_output.columns) & set(skill_input.columns)
                skill_input_reduced = skill_input.drop(columns=cols_to_drop)

                skill_input = skill_input_reduced.merge(
                    skill_output, left_index=True, right_index=True, how="inner"
                )
            elif isinstance(skill, (AnalysisSkill, SynthesisSkill)):
                skill_input = skill_output
            else:
                raise ValueError(f"Unsupported skill type: {type(skill)}")
        if isinstance(skill_input, InternalSeries):
            skill_input = skill_input.to_frame().T
        return skill_input

    def __rich__(self):
        """Returns a rich representation of the skill."""
        # TODO: move it to a base class and use repr derived from Skills
        text = f"[bold blue]Total Agent Skills: {len(self.skills)}[/bold blue]\n\n"
        for skill in self.skills.values():
            text += (
                f"[bold underline green]{skill.name}[/bold underline green]\n"
                f"[green]{skill.instructions}[green]\n"
            )
        return text


class ParallelSkillSet(SkillSet):
    """
    Represents a set of skills that are acquired simultaneously to reach a goal.

    In a ParallelSkillSet, each skill can be developed independently of the others. This is useful
    for agents that require multiple, diverse capabilities, or tasks where each skill contributes a piece of
    the overall solution.

    Examples:
        Create a ParallelSkillSet with a list of skills specified as BaseSkill instances
        >>> from adala.skills import ParallelSkillSet, ClassificationSkill, TransformSkill
        >>> skillset = ParallelSkillSet(skills=[ClassificationSkill(), TransformSkill()])
    """

    def apply(
        self,
        input: Union[InternalSeries, InternalDataFrame],
        runtime: Runtime,
        improved_skill: Optional[str] = None,
    ) -> InternalDataFrame:
        """
        Applies each skill on the dataset, enhancing the agent's experience.

        Args:
            input (Union[Record, InternalDataFrame]): Input data
            runtime (Runtime): The runtime environment in which to apply the skills.
            improved_skill (Optional[str], optional): Unused in ParallelSkillSet. Defaults to None.
        Returns:
            Union[Record, InternalDataFrame]: Skill predictions.
        """
        if improved_skill:
            # start from the specified skill, assuming previous skills have already been applied
            skill_sequence = [improved_skill]
        else:
            skill_sequence = list(self.skills.keys())

        skill_outputs = []
        for i, skill_name in enumerate(skill_sequence):
            skill = self.skills[skill_name]
            # use input dataset for the first node in the pipeline
            print_text(f"Applying skill: {skill_name}")
            skill_output = skill.apply(input, runtime)
            skill_outputs.append(skill_output)
        if not skill_outputs:
            return InternalDataFrame()
        else:
            if isinstance(skill_outputs[0], InternalDataFrame):
                skill_outputs = InternalDataFrameConcat(skill_outputs, axis=1)
                cols_to_drop = set(input.columns) & set(skill_outputs.columns)
                skill_input_reduced = input.drop(columns=cols_to_drop)

                return skill_input_reduced.merge(
                    skill_outputs, left_index=True, right_index=True, how="inner"
                )
            elif isinstance(skill_outputs[0], (dict, InternalSeries)):
                # concatenate output to each row of input
                output = skill_outputs[0]
                return InternalDataFrameConcat(
                    [
                        input,
                        InternalDataFrame(
                            [output] * len(input),
                            columns=output.index,
                            index=input.index,
                        ),
                    ],
                    axis=1,
                )
            else:
                raise ValueError(f"Unsupported output type: {type(skill_outputs[0])}")
