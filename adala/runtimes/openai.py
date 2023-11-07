import os
import openai
from pydantic import model_validator, field_validator, ValidationInfo, Field
from typing import Optional, Dict
from .base import LLMRuntime, LLMRuntimeType, LLMRuntimeModelType
from adala.utils.logs import print_error


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
    
    api_key: Optional[str] = None
    gpt_model_name: Optional[str] = Field(default='gpt-3.5-turbo-instruct', alias='model')
    temperature: Optional[float] = 0.0

    def _check_api_key(self):
        if self.api_key:
            return
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print_error(
                'OpenAI API key is not provided. Please set the OPENAI_API_KEY environment variable:\n\n'
                'export OPENAI_API_KEY=your-openai-api-key\n\n'
                'or set the `api_key` attribute of the `OpenAIRuntime` python class:\n\n'
                f'{self.__class__.__name__}(..., api_key="your-openai-api-key")\n\n'
                f'Read more about OpenAI API keys at https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key')
            raise ValueError('OpenAI API key is not provided.')

    def _check_model_availability(self):
        models = openai.Model.list(api_key=self.api_key)
        models = set(model['id'] for model in models['data'])
        if self.gpt_model_name not in models:
            print_error(
                f'Requested model "{self.gpt_model_name}" is not available in your OpenAI account. '
                f'Available models are: {models}\n\n'
                f'Try to change the runtime settings for {self.__class__.__name__}, for example:\n\n'
                f'{self.__class__.__name__}(..., model="gpt-3.5-turbo")\n\n'
            )
            raise ValueError(f'Requested model {self.gpt_model_name} is not available in your OpenAI account.')

    def init_runtime(self):
        self._check_api_key()
        self._check_model_availability()

        student_models = {'gpt-3.5-turbo-instruct', 'text-davinci-003'}
        teacher_models = {'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4-1106-preview', 'gpt-4-vision-preview'}

        if self.gpt_model_name in student_models:
            self.llm_runtime_type = LLMRuntimeType.STUDENT
        elif self.gpt_model_name in teacher_models:
            self.llm_runtime_type = LLMRuntimeType.TEACHER
        else:
            raise NotImplementedError(f'Not supported model: {self.gpt_model_name}.')

        self.llm_runtime_model_type = LLMRuntimeModelType.OpenAI
        self.llm_params = {
            'model': self.gpt_model_name,
            'temperature': self.temperature,
            'api_key': self.api_key
        }
        self._create_program()
        return self
