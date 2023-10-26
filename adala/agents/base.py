from pydantic import BaseModel, Field, SkipValidation, field_validator, model_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union
from adala.environments.base import Environment, BasicEnvironment
from adala.datasets import Dataset, DataFrameDataset
from adala.runtimes.base import Runtime, LLMRuntime, LLMRuntimeType, LLMRuntimeModelType
from adala.runtimes.openai import OpenAIRuntime
from adala.memories.base import ShortTermMemory, LongTermMemory
from adala.skills.base import BaseSkill
from adala.skills.skillset import SkillSet, LinearSkillSet
from adala.utils.logs import print_dataframe, print_text, print_error
from adala.utils.internal_data import InternalDataFrame


class Agent(BaseModel, ABC):
    """
    Represents a customizable agent that can interact with environments, 
    employ skills, and leverage memory and runtimes.

    Attributes:
        environment (Union[Dataset, Environment]): The environment with which the agent interacts.
        skills (Union[SkillSet, BaseSkill, List[BaseSkill], Dict[str, BaseSkill]]): The skills possessed by the agent.
        memory (LongTermMemory, optional): The agent's long-term memory. Defaults to None.
        runtimes (Dict[str, Runtime], optional): The runtimes available to the agent. Defaults to predefined runtimes.
        default_runtime (str): The default runtime used by the agent. Defaults to 'openai'.
    """
    
    environment: Union[InternalDataFrame, Dataset, Environment] = Field(default_factory=DataFrameDataset)
    skills: Union[SkillSet, BaseSkill, List[BaseSkill], Dict[str, BaseSkill]]

    memory: LongTermMemory = Field(default=None)
    runtimes: Optional[Dict[str, Runtime]] = Field(
        default_factory=lambda: {
            'openai': OpenAIRuntime(model='gpt-3.5-turbo-instruct'),
            # 'llama2': LLMRuntime(
            #     llm_runtime_type=LLMRuntimeModelType.Transformers,
            #     llm_params={
            #         'model': 'meta-llama/Llama-2-7b',
            #         'device': 'cuda:0',
            #     }
            # )
        }
    )
    teacher_runtimes: Optional[Dict[str, Runtime]] = Field(
        default_factory=lambda: {
            'openai-gpt3': OpenAIRuntime(model='gpt-3.5-turbo'),
            # 'openai-gpt4': OpenAIRuntime(model='gpt-4')
        }
    )
    default_runtime: str = 'openai'
    default_teacher_runtime: str = 'openai-gpt3'

    class Config:
        arbitrary_types_allowed = True

    def __rich__(self):
        """
        Returns a colorized and formatted representation of the Agent instance.

        Returns:
            str: A rich-formatted representation of the agent.
        """
        
        skill_names = ", ".join([skill.name for skill in self.skills.skills.values()])
        runtime_names = ", ".join(self.runtimes.keys())
        
        return (
            f"[bold blue]Agent Instance[/bold blue]\n\n"
            f"Environment: {self.environment.__class__.__name__}\n"
            f"Skills: {skill_names}\n"
            f"Runtimes: {runtime_names}\n"
            f"Default Runtime: {self.default_runtime}\n"
            f"Default Teacher Runtime: {self.default_teacher_runtime}"
        )

    @field_validator('environment')
    def environment_validator(cls, v):
        """
        Validates and possibly transforms the environment attribute.

        Args:
            v (Union[Dataset, Environment]): The environment value to validate.

        Returns:
            Environment: The validated environment.
        """
        if isinstance(v, InternalDataFrame):
            v = DataFrameDataset(df=v)
        if isinstance(v, Dataset):
            v = BasicEnvironment(dataset=v)
        return v

    @field_validator('skills')
    def skills_validator(cls, v):
        """
        Validates and possibly transforms the skills attribute.

        Args:
            v (Union[SkillSet, BaseSkill, List[BaseSkill], Dict[str, BaseSkill]]): The skills value to validate.

        Returns:
            SkillSet: The validated set of skills.
        """
        
        if isinstance(v, SkillSet):
            pass
        elif isinstance(v, BaseSkill):
            v = LinearSkillSet(skills={'skill_0': v})
        elif isinstance(v, list):
            v = LinearSkillSet(skills={f'skill_{i}': skill for i, skill in enumerate(v)})
        elif isinstance(v, dict):
            v = LinearSkillSet(skills=v)
        return v

    @model_validator(mode='after')
    def verify_input_parameters(self):
        def _raise_default_runtime_error(val, runtime, runtimes, default_value):
            print_error(f"The Agent.{runtime} is set to {val}, "
                        f"but this runtime is not available in the list: {list(runtimes)}. "
                        f"Please choose one of the available runtimes and initialize the agent again, for example:\n\n"
                        f"agent = Agent(..., {runtime}='{default_value}')\n\n"
                        f"Make sure the default runtime is available in the list of runtimes. For example:\n\n"
                        f"agent = Agent(..., runtimes={{'{default_value}': OpenAIRuntime(model='gpt-4')}})\n\n")
            raise ValueError(f"default runtime {val} not found in provided runtimes.")

        if self.default_runtime not in self.runtimes:
            _raise_default_runtime_error(self.default_runtime, 'default_runtime', self.runtimes, 'openai')
        if self.default_teacher_runtime not in self.teacher_runtimes:
            _raise_default_runtime_error(self.default_teacher_runtime, 'default_teacher_runtime', self.teacher_runtimes, 'openai-gpt4')
        return self

    def get_runtime(self, runtime: Optional[str] = None) -> Runtime:
        """
        Retrieves the specified runtime or the default runtime if none is specified.

        Args:
            runtime (str, optional): The name of the runtime to retrieve. Defaults to None.

        Returns:
            Runtime: The requested runtime.

        Raises:
            ValueError: If the specified runtime is not found.
        """
        
        if runtime is None:
            runtime = self.default_runtime
        if runtime not in self.runtimes:
            raise ValueError(f'Runtime "{runtime}" not found.')
        return self.runtimes[runtime]

    def get_teacher_runtime(self, runtime: Optional[str] = None) -> Runtime:
        """
        Retrieves the specified teacher runtime or the default runtime if none is specified.

        Args:
            runtime (str, optional): The name of the runtime to retrieve. Defaults to None.

        Returns:
            Runtime: The requested runtime.

        Raises:
            ValueError: If the specified runtime is not found.
        """

        if runtime is None:
            runtime = self.default_teacher_runtime
        if runtime not in self.teacher_runtimes:
            raise ValueError(f'Teacher Runtime "{runtime}" not found.')
        return self.teacher_runtimes[runtime]

    def apply_skills(
        self,
        dataset: Union[Dataset, InternalDataFrame],
        runtime: Optional[Union[str, Runtime]] = None,
        experience: Optional[ShortTermMemory] = None,
    ) -> ShortTermMemory:
        """
        Applies the agent's skills to a given dataset using the specified runtime.

        Args:
            dataset (Dataset): The dataset to apply skills on.
            runtime (str, optional): The runtime to use. Defaults to None.
            experience (ShortTermMemory, optional): The agent's short-term memory. Defaults to None.

        Returns:
            ShortTermMemory: The short-term memory resulting from the application of skills.
        """
        runtime = runtime or self.default_runtime
        if isinstance(dataset, InternalDataFrame):
            dataset = DataFrameDataset(df=dataset)
        if isinstance(runtime, str):
            runtime = self.get_runtime(runtime=runtime)
        return self.skills.apply(dataset=dataset, runtime=runtime, experience=experience)

    def learn(
        self,
        learning_iterations: int = 3,
        accuracy_threshold: float = 0.9,
        update_skills: bool = True,
        update_memory: bool = True,
        request_environment_feedback: bool = True,
        experience: Optional[ShortTermMemory] = None,
        runtime: Optional[str] = None,
    ) -> ShortTermMemory:
        """
        Enables the agent to learn and improve its skills based on interactions with its environment.

        Args:
            learning_iterations (int, optional): The number of iterations for learning. Defaults to 3.
            accuracy_threshold (float, optional): The desired accuracy threshold to reach. Defaults to 0.9.
            update_skills (bool, optional): Flag to determine if skills should be updated after learning. Defaults to True.
            update_memory (bool, optional): Flag to determine if memory should be updated after learning. Defaults to True.
            request_environment_feedback (bool, optional): Flag to determine if feedback should be requested from the environment. Defaults to True.
            experience (ShortTermMemory, optional): Initial experience for the learning process. Defaults to None.
            runtime (str, optional): The runtime to be used for the learning process. Defaults to None.

        Returns:
            ShortTermMemory: The short-term memory after the learning process.
        """
        
        runtime = self.get_runtime(runtime=runtime)
        # TODO: support teacher runtime input, not default
        teacher_runtime = self.get_teacher_runtime(runtime=self.default_teacher_runtime)

        skills = self.skills.model_copy(deep=True)
        dataset = self.environment.as_dataset()

        # Apply agent skills to dataset and get experience with predictions
        experience = self.apply_skills(dataset=dataset, runtime=runtime, experience=experience)

        # Agent select one skill to improve
        learned_skill = skills.select_skill_to_improve(experience)

        # Request feedback from environment is necessary
        if request_environment_feedback:
            self.environment.request_feedback(learned_skill, experience)

        for iteration in range(learning_iterations):
            print_text(f'\n\n=> Iteration #{iteration}: Comparing to ground truth, analyzing and improving ...')

            # 1. EVALUATION PHASE: Compare predictions to ground truth
            experience = self.environment.compare_to_ground_truth(learned_skill, experience)
            print_text(f'Comparing predictions to ground truth data ...')
            print_dataframe(experience.evaluations)

            # 2. ANALYSIS PHASE: Analyze evaluation experience, optionally use long term memory
            print_text(f'Analyze evaluation experience ...')
            experience = learned_skill.analyze(
                experience=experience,
                student_runtime=runtime,
                teacher_runtime=teacher_runtime,
                memory=self.memory
            )
            print_text(f'Number of errors: {len(experience.errors)}')

            print_text(f'Accuracy = {experience.accuracy*100:0.2f}%', style='bold red')
            if experience.accuracy >= accuracy_threshold:
                print_text(f'Accuracy threshold reached ({experience.accuracy} >= {accuracy_threshold})')
                break

            # 3. IMPROVEMENT PHASE: Improve skills based on analysis
            print_text(f"Improve \"{learned_skill.name}\" skill based on analysis ...")
            experience = learned_skill.improve(
                experience=experience,
                runtime=teacher_runtime,
                update_instructions=True
            )
            print_text(f'Updated instructions for skill "{learned_skill.name}":\n')
            print_text(learned_skill.instructions, style='bold green')

            # 4. RE-APPLY PHASE: Re-apply skills to dataset
            print_text(f"Re-apply {learned_skill.name} skill to dataset ...")
            experience = learned_skill.apply(dataset, runtime, experience=experience)

        # Mark skill as evolved - it will not be selected for improvement again
        learned_skill.evolved = True

        # Update skills and memory based on experience
        if update_skills:
            self.skills = skills

        if self.memory and update_memory:
            self.memory.remember(experience, self.skills)

        print_text('Train is done!')
        return experience
