from pydantic import model_validator
from .base import LLMRuntime, LLMRuntimeModelType


class OpenAIRuntime(LLMRuntime):
    """Runtime class specifically designed for OpenAI models.

    This class is tailored to use OpenAI models, particularly GPT models.
    It inherits from the `LLMRuntime` class and thus can utilize its functionalities but specializes 
    for the OpenAI ecosystem.

    Attributes:
        api_key (str): The API key required to access OpenAI's API.
        gpt_model_name (str): Name of the GPT model. Defaults to 'gpt-3.5-turbo-instruct'.
        temperature (float): Sampling temperature for the GPT model's output. 
                             A higher value makes output more random, while a lower value makes it more deterministic.
                             Defaults to 0.0.
    """
    
    api_key: str
    gpt_model_name: str = 'gpt-3.5-turbo-instruct'
    temperature: float = 0.0

    @model_validator(mode='after')
    def init_runtime(self):
        """Initializes the OpenAI runtime environment.

        Configures the LLM runtime type to OpenAI and sets LLM parameters based on class attributes.

        Returns:
            OpenAIRuntime: Initialized OpenAI runtime instance.
        """
        
        self.llm_runtime_type = LLMRuntimeModelType.OpenAI
        self.llm_params = {
            'model': self.gpt_model_name,
            'temperature': self.temperature,
            'api_key': self.api_key,
        }
        return self
