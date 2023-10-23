import pandas as pd

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
    Base class for agents.
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
                }
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

    @field_validator('environment')
    def environment_validator(cls, v):
        if isinstance(v, Dataset):
            v = Environment(dataset=v)
        return v

    @field_validator('skills')
    def skills_validator(cls, v):
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
        if runtime is None:
            runtime = self.default_runtime
        if runtime not in self.runtimes:
            raise ValueError(f'Runtime "{runtime}" not found.')
        return self.runtimes[runtime]

    def apply_skills(self, dataset: Dataset, runtime: Optional[str] = None) -> ShortTermMemory:
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

        runtime = self.get_runtime(runtime=runtime)

        skills = self.skills.model_copy(deep=True)

        # Apply agent skills to dataset and get experience with predictions
        experience = skills.apply(dataset=self.environment.ground_truth_dataset, runtime=runtime, experience=experience)

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
            experience = learned_skill.apply(train_dataset, runtime, experience=experience)

        # Update skills and memory based on experience
        if update_skills:
            self.skills = skills

        if self.memory and update_memory:
            self.memory.remember(experience, self.skills)

        log('Done!')

        return experience
