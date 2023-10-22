from pydantic import BaseModel, Field, SkipValidation, validator
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from adala.envs.base import BaseEnvironment, StaticEnvironment
from adala.datasets.base import Dataset, BlankDataset
from adala.runtimes.base import Runtime, LLMRuntime, LLMRuntimeModelType
from adala.memories.base import ShortTermMemory, LongTermMemory

# following the protocol https://agentprotocol.ai/protocol


class AgentStep(BaseModel):
    """
    Base class for agent steps results
    """
    experience: ShortTermMemory
    is_last: bool


class Agent(BaseModel, ABC):
    """
    Base class for agents.
    """
    dataset: Optional[Dataset] # = Field(default_factory=lambda: BlankDataset())
    env: Optional[Environment] # = Field(default_factory=lambda: BlankEnvironment())
    
    memory: Optional[LongTermMemory] = Field(default=None)
    
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

    @validator('env', pre=True, always=True)
    def set_defaults(cls, env, values):
        if env is None:
            if 'dataset' in values and values['dataset'] is not None:
                return StaticEnvironment(dataset=values['dataset'])
            else:
                raise ValueError("Either dataset or env must be initialized.")

        return env
    
    @abstractmethod
    def greet(self) -> str:
        """
        Return agent greeting and description
        """

    @abstractmethod
    def run(self) -> AgentStep:
        """
        Run agent and return results
        """

    @abstractmethod
    def learn(self) -> AgentStep:
        """
        Learn from dataset and return results
        """
