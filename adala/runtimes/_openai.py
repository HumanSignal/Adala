import os
from typing import Any, Dict, List, Optional

from adala.utils.matching import match_options
from adala.utils.parse import parse_template, partial_str_format
from openai import NotFoundError, OpenAI
from pydantic import ConfigDict, Field, computed_field
from rich import print

from .base import Runtime
from ._litellm import (
    LiteLLMChatRuntime,
    AsyncLiteLLMChatRuntime,
    LiteLLMVisionRuntime,
)


class OpenAIChatRuntime(LiteLLMChatRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion models to perform the skill.

    Attributes:
        model: OpenAI model name.
        openai_api_key: OpenAI API key. If not provided, will be taken from OPENAI_API_KEY environment variable.
        base_url: Can point to any implementation of the OpenAI API. Defaults to OpenAI's.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
    """

    model: str
    # openai_model: str = Field(alias='model')
    # TODO does it make any sense for this to be optional?
    api_key: Optional[str] = Field(
        default=os.getenv('OPENAI_API_KEY'), alias='openai_api_key'
    )


class AsyncOpenAIChatRuntime(AsyncLiteLLMChatRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion models to perform the skill.
    It uses async calls to OpenAI API.

    Attributes:
        model: OpenAI model name.
        openai_api_key: OpenAI API key. If not provided, will be taken from OPENAI_API_KEY environment variable.
        base_url: Can point to any implementation of the OpenAI API. Defaults to OpenAI's.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        temperature: Temperature for sampling, between 0 and 1. Higher values means the model will take more risks.
            Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            Defaults to 0.0.

        concurrent_clients: Number of concurrent clients to OpenAI API. More clients means more parallel requests, but
            also more money spent and more chances to hit the rate limit. Defaults to 10.
    """

    model: str
    api_key: Optional[str] = Field(
        default=os.getenv('OPENAI_API_KEY'), alias='openai_api_key'
    )


class OpenAIVisionRuntime(LiteLLMVisionRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and vision models to perform the skill.
    Only compatible with OpenAI API version 1.0.0 or higher.
    """

    model: str
    api_key: Optional[str] = Field(
        default=os.getenv('OPENAI_API_KEY'), alias='openai_api_key'
    )
    # NOTE this check used to exist in OpenAIVisionRuntime.record_to_record,
    #      but doesn't seem to have a definition
    # def init_runtime(self) -> 'Runtime':
    #     if not check_if_new_openai_version():
    #         raise NotImplementedError(
    #             f'{self.__class__.__name__} requires OpenAI API version 1.0.0 or higher.'
    #         )
    #     super().init_runtime()
