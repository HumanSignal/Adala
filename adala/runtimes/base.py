from abc import ABC, abstractmethod
from pydantic import BaseModel, root_validator
from typing import List


class Runtime(BaseModel, ABC):
    """
    Base class for runtimes.
    """


class LLMRuntime(Runtime):
    """
    Base class for LLM runtimes.
    """
    @root_validator
    def check_runtime(cls, values):
        """
        Check that runtime is valid.
        Use this method to initialize runtime.
        """
        return values

    @abstractmethod
    def process_batch(self, batch: List[str]) -> List[str]:
        """
        Process batch of data and return results.
        """
        pass


class CodeRuntime(Runtime):
    """
    Base class for code runtimes.
    """
