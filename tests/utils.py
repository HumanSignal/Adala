import enum
from dataclasses import dataclass
from unittest.mock import patch
from typing import List
from pydantic import field_validator


class PatchedCalls(enum.Enum):
    GUIDANCE = "guidance._program.Program.__call__"
    # OPENAI_MODEL_LIST = 'openai.models.list'
    OPENAI_MODEL_RETRIEVE = "openai.resources.models.Models.retrieve"
    OPENAI_CHAT_COMPLETION = "openai.resources.chat.completions.Completions.create"
    OPENAI_EMBEDDING_CREATE = "openai.resources.embeddings.Embeddings.create"


@dataclass
class OpenaiChatCompletionMessageMock(object):
    content: str


@dataclass
class OpenaiChatCompletionChoiceMock(object):
    message: OpenaiChatCompletionMessageMock


class OpenaiChatCompletionMock(object):
    """
    Currently, OpenAI python client expects the following API response structure:

    completion.choices[0].message.content

    """

    def __init__(self, content):
        self.choices = [
            OpenaiChatCompletionChoiceMock(
                message=OpenaiChatCompletionMessageMock(content=content)
            )
        ]


def patching(target_function, data, strict=False):
    """
    A decorator that patches the specified function, making it return the expected output for the given input.

    Args:
    - target_function (str): The function to patch, in 'module.Class.method' format.
    - data (list of dict): A list containing dictionaries with 'input' and 'output' keys.

    Example:
    @patching(target_function='internal.function', data=[{"input": {"arg_0": "first argument", "any": "input kwargs"}, "output": <some expected output object>}]
    def test_my_function():
        my_function()
    """

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            call_index = [0]  # Using list to make it mutable inside nested function

            def side_effect(*args, **kwargs):
                if call_index[0] >= len(data):
                    raise AssertionError(
                        f"Unexpected call number {call_index[0]} to {target_function}"
                    )

                expected_input = data[call_index[0]]["input"]
                expected_output = data[call_index[0]]["output"]

                # Merging positional arguments into the keyword arguments for comparison
                actual_input = {**kwargs}
                for i, value in enumerate(args):
                    key = f"arg_{i}"
                    actual_input[key] = value

                if strict:
                    if actual_input != expected_input:
                        raise AssertionError(
                            f"Expected input {expected_input}\n\n"
                            f"but got {actual_input}\non call number {call_index[0]}"
                            f" to {target_function}"
                        )
                else:
                    for key, value in expected_input.items():
                        if key not in actual_input:
                            raise AssertionError(
                                f"Expected input {expected_input}\n\n"
                                f"but key '{key}' was missing "
                                f"on actual call number {call_index[0]} "
                                f"to {target_function}.\n\n"
                                f"Actual input: {actual_input}"
                            )
                        if actual_input[key] != value:
                            raise AssertionError(
                                f"Expected input {expected_input}\n\n"
                                f"but actual_input['{key}'] != expected_input['{key}']\n"
                                f"on call number {call_index[0]} "
                                f"to {target_function}.\n\n"
                                f"Actual input: {actual_input}"
                            )

                call_index[0] += 1
                return expected_output

            with patch(target_function, side_effect=side_effect):
                result = test_func(*args, **kwargs)
                if call_index[0] != len(data):
                    raise AssertionError(
                        f"Expected {len(data)} calls to {target_function}, but got {call_index[0]}"
                    )
                return result

        return wrapper

    return decorator


class mdict(dict):
    def __getattr__(self, item):
        return self[item]
