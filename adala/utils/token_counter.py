"""
Token counter implementation for calculating token usage.

The main interface includes:
- TokenCounter.count_tokens() -> int

Other methods are used internally and not intended for external use.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import litellm
from litellm.utils import token_counter

logger = logging.getLogger(__name__)


class TokenCounter(ABC):
    """
    Abstract base class for token counting.

    This class defines the interface for token counting implementations.
    Concrete implementations must provide a `count_tokens` method.
    """

    @abstractmethod
    def count_tokens(self, messages: List[Dict[str, Any]], model: str) -> int:
        """Count tokens for given messages and model"""

    @abstractmethod
    def max_tokens(self, model: str) -> int | None:
        """Get the maximum number of tokens for a given model or None if the model is not supported"""


class LiteLLMTokenCounter(TokenCounter):
    """
    Concrete implementation of token counting using the LiteLLM library.

    This class provides a concrete implementation of the `TokenCounter` interface
    using the LiteLLM library. It counts tokens for given messages and model.
    """

    def count_tokens(self, messages: List[Dict[str, Any]], model: str) -> int:
        return token_counter(model=model, messages=messages)

    def max_tokens(self, model: str) -> int | None:
        if model not in litellm.model_cost:
            logger.warning(
                "Model '%s' not found in litellm.model_cost. Returning None.", model
            )
            return None

        return litellm.model_cost[model].get(
            "max_input_tokens", litellm.model_cost[model].get("max_tokens")
        )


def get_token_counter(name: str = "litellm") -> TokenCounter:
    """
    Factory function to get a token counter implementation by name.

    Args:
        name: The name of the token counter implementation to use.
            Currently supported values:
            - "litellm": Uses the LiteLLM token counter

    Returns:
        A concrete implementation of TokenCounter

    Raises:
        ValueError: If the requested token counter implementation is not supported
    """
    token_counters = {
        "litellm": LiteLLMTokenCounter,
    }

    if name not in token_counters:
        raise ValueError(
            f"Token counter '{name}' not supported. Available options: {list(token_counters.keys())}"
        )

    return token_counters[name]()
