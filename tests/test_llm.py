import pytest
import asyncio
import pandas as pd
from adala.runtimes import LiteLLMChatRuntime, AsyncLiteLLMChatRuntime


@pytest.mark.vcr
def test_llm_sync():

    runtime = LiteLLMChatRuntime()

    # test plaintext success

    result = runtime.get_llm_response(
        messages=[{'role': 'user', 'content': 'return the word Banana with exclamation mark'}],
    )
    expected_result = "Banana!"
    assert result == expected_result

    # test structured success

    result = runtime.record_to_record(
        record={'input_name': 'Carla', 'input_age': 25},
        input_template="My name is {input_name} and I am {input_age} years old.",
        instructions_template="",
        output_template="name: {name}, age: {age}",
    )

    # note age coerced to string
    expected_result = {'name': 'Carla', 'age': '25'}
    assert result == expected_result


@pytest.mark.vcr
def test_llm_async():

    runtime = AsyncLiteLLMChatRuntime()

    batch = pd.DataFrame.from_records([{'input_name': 'Carla', 'input_age': 25}])

    result = asyncio.run(
        runtime.batch_to_batch(
            batch,
            input_template="My name is {input_name} and I am {input_age} years old.",
            instructions_template="",
            output_template="name: {name}, age: {age}",
        )
    )

    # note age coerced to string
    expected_result = pd.DataFrame.from_records([{'name': 'Carla', 'age': '25'}])
    # need 2 all() for row and column axis
    assert (result == expected_result).all().all()
