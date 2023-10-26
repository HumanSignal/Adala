import enum
from unittest.mock import patch


class PatchedCalls(enum.Enum):
    GUIDANCE = 'guidance._program.Program.__call__'
    OPENAI_MODEL_LIST = 'openai.api_resources.model.Model.list'


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
                    raise AssertionError(f"Unexpected call number {call_index[0] + 1} to {target_function}")

                expected_input = data[call_index[0]]['input']
                expected_output = data[call_index[0]]['output']

                # Merging positional arguments into the keyword arguments for comparison
                actual_input = {**kwargs}
                for i, value in enumerate(args):
                    key = f"arg_{i}"
                    actual_input[key] = value

                if strict:
                    if actual_input != expected_input:
                        raise AssertionError(
                            f"Expected input {expected_input}\n\nbut got {actual_input}\non call number {call_index[0] + 1} to {target_function}")
                else:
                    for key, value in expected_input.items():
                        if key not in actual_input:
                            raise AssertionError(
                                f"Expected input {expected_input}\n\nbut key '{key}' was missing on actual call number {call_index[0] + 1} to {target_function}.\n\nActual input: {actual_input}")
                        if actual_input[key] != value:
                            raise AssertionError(
                                f"Expected input {expected_input}\n\nbut actual_input['{key}'] != expected_input['{key}']\non call number {call_index[0] + 1} to {target_function}.\n\nActual input: {actual_input}")

                call_index[0] += 1
                return expected_output

            with patch(target_function, side_effect=side_effect):
                return test_func(*args, **kwargs)

        return wrapper

    return decorator
