import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pydantic import BaseModel, Field, SkipValidation, field_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union
from adala.environments.base import Environment
from adala.datasets import Dataset, DataFrameDataset
from adala.runtimes.base import Runtime, LLMRuntime, LLMRuntimeModelType
from adala.memories.base import ShortTermMemory, LongTermMemory
from adala.skills.base import BaseSkill
from adala.skills.skillset import SkillSet, LinearSkillSet
from adala.utils.logs import log


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
    
    environment: Union[Dataset, Environment]
    skills: Union[SkillSet, BaseSkill, List[BaseSkill], Dict[str, BaseSkill]]

    memory: LongTermMemory = Field(default=None)
    runtimes: Optional[Dict[str, Runtime]] = Field(
        default_factory=lambda: {
            'openai': LLMRuntime(
                llm_runtime_type=LLMRuntimeModelType.OpenAI,
                llm_params={
                    'model': 'gpt-3.5-turbo-instruct',
                },
                # verbose=True
            ),
            'openai-gpt4': LLMRuntime(
                llm_runtime_type=LLMRuntimeModelType.OpenAI,
                llm_params={
                    'model': 'gpt-4',
                }
            ),
            # 'llama2': LLMRuntime(
            #     llm_runtime_type=LLMRuntimeModelType.Transformers,
            #     llm_params={
            #         'model': 'meta-llama/Llama-2-7b',
            #         'device': 'cuda:0',
            #     }
            # )
        }
    )
    default_runtime: str = 'openai'

    class Config:
        arbitrary_types_allowed = True

    def __rich__(self):
        """
        Returns a colorized and formatted representation of the Agent instance.

        Returns:
            str: A rich-formatted representation of the agent.
        """
        
        skill_names = ", ".join(self.skills.skills.keys())
        runtime_names = ", ".join(self.runtimes.keys())
        
        return (
            f"[bold blue]Agent Instance[/bold blue]\n\n"
            f"Environment: {self.environment.__class__.__name__}\n"
            f"Skills: {skill_names}\n"
            f"Runtimes: {runtime_names}\n"
            f"Default Runtime: {self.default_runtime}"
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
        
        if isinstance(v, Dataset):
            v = Environment(dataset=v)
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

    def apply_skills(self, dataset: Dataset, runtime: Optional[str] = None) -> ShortTermMemory:
        """
        Applies the agent's skills to a given dataset using the specified runtime.

        Args:
            dataset (Dataset): The dataset to apply skills on.
            runtime (str, optional): The runtime to use. Defaults to None.

        Returns:
            ShortTermMemory: The short-term memory resulting from the application of skills.
        """
        
        return self.skills.apply(dataset=dataset, runtime=self.get_runtime(runtime=runtime))

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

        skills = self.skills.model_copy(deep=True)
        dataset = self.environment.as_dataset()

        # Apply agent skills to dataset and get experience with predictions
        experience = skills.apply(dataset=dataset, runtime=runtime, experience=experience)

        # Agent select one skill to improve
        learned_skill = skills.select_skill_to_improve(experience)

        # Request feedback from environment is necessary
        if request_environment_feedback:
            self.environment.request_feedback(learned_skill, experience)

        for iteration in range(learning_iterations):

            log(f'Iteration #{iteration}: Comparing to ground truth, analyzing and improving...')

            # 1. EVALUATION PHASE: Compare predictions to ground truth
            experience = self.environment.compare_to_ground_truth(learned_skill, experience)

            # 2. ANALYSIS PHASE: Analyze evaluation experience, optionally use long term memory
            experience = learned_skill.analyze(experience, self.memory, runtime)

            if experience.accuracy >= accuracy_threshold:
                log(f'Accuracy threshold reached ({experience.accuracy} >= {accuracy_threshold})')
                break

            # 3. IMPROVEMENT PHASE: Improve skills based on analysis
            experience = learned_skill.improve(experience)

            # 4. RE-APPLY PHASE: Re-apply skills to dataset
            experience = learned_skill.apply(dataset, runtime, experience=experience)

        # Update skills and memory based on experience
        if update_skills:
            self.skills = skills

        if self.memory and update_memory:
            self.memory.remember(experience, self.skills)

        log('Done!')

        return experience
