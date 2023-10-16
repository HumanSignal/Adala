from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, Dict


class Runtime(BaseModel, ABC):
    """
    Base class for runtimes.
    """
    verbose: bool = False

    @model_validator(mode='after')
    def init_runtime(self):
        """
        Check that runtime is valid.
        Use this method to initialize runtime.
        """
        return self


class LLMRuntime(Runtime):
    """
    Base class for LLM runtimes.
    """
    text_key: str = 'text'
    score_key: str = 'score'
    rationale_key: str = 'rationale'

    @abstractmethod
    def process_batch(self, batch: List[str]) -> List[Dict]:
        """
        Process batch of data and return results in the same order as input data.
        """
        pass


class CodeRuntime(Runtime):
    """
    Base class for code runtimes.
    """
