from pydantic import model_validator
from .base import LLMRuntime, LLMRuntimeModelType


class OpenAIRuntime(LLMRuntime):
    """
    OpenAI runtime.
    """
    api_key: str
    gpt_model_name: str = 'gpt-3.5-turbo-instruct'
    temperature: float = 0.0

    @model_validator(mode='after')
    def init_runtime(self):
        self.llm_runtime_type = LLMRuntimeModelType.OpenAI
        self.llm_params = {
            'model': self.gpt_model_name,
            'temperature': self.temperature,
            'api_key': self.api_key,
        }
        return self
