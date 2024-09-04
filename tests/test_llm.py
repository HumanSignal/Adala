import pytest
import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from adala.runtimes import LiteLLMChatRuntime, AsyncLiteLLMChatRuntime


@pytest.mark.vcr
def test_llm_sync():
    runtime = LiteLLMChatRuntime()

    # test plaintext success

    result = runtime.get_llm_response(
        messages=[
            {"role": "user", "content": "return the word Banana with exclamation mark"}
        ],
    )
    expected_result = "Banana!"
    assert result == expected_result

    # test structured success

    class Output(BaseModel):
        name: str = Field(..., description="name:")
        age: str = Field(..., description="age:")

    result = runtime.record_to_record(
        record={"input_name": "Carla", "input_age": 25},
        input_template="My name is {input_name} and I am {input_age} years old.",
        instructions_template="",
        response_model=Output,
    )

    # note age coerced to string
    expected_result = {
        "name": "Carla",
        "age": "25",
        "_prompt_tokens": 86,
        "_completion_tokens": 10,
        "_prompt_cost_usd": 1.29e-05,
        "_completion_cost_usd": 6e-06,
        "_total_cost_usd": 1.89e-05,
    }
    assert result == expected_result

    # test structured failure

    runtime.api_key = "fake_api_key"

    result = runtime.record_to_record(
        record={"input_name": "Carla", "input_age": 25},
        input_template="My name is {input_name} and I am {input_age} years old.",
        instructions_template="",
        response_model=Output,
    )

    expected_result = {
        "_adala_error": True,
        "_adala_message": "AuthenticationError",
        "_adala_details": "litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: fake_api_key. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
        "_prompt_tokens": 9,
        "_completion_tokens": 0,
        "_prompt_cost_usd": 1.35e-06,
        "_completion_cost_usd": 0.0,
        "_total_cost_usd": 1.35e-06,
    }
    assert result == expected_result


@pytest.mark.vcr
def test_llm_async():
    # test success

    runtime = AsyncLiteLLMChatRuntime()

    batch = pd.DataFrame.from_records([{"input_name": "Carla", "input_age": 25}])

    class Output(BaseModel):
        name: str = Field(..., description="name:")
        age: str = Field(..., description="age:")

    result = asyncio.run(
        runtime.batch_to_batch(
            batch,
            input_template="My name is {input_name} and I am {input_age} years old.",
            instructions_template="",
            response_model=Output,
        )
    )

    # note age coerced to string
    expected_result = pd.DataFrame.from_records(
        [
            {
                "name": "Carla",
                "age": "25",
                "_prompt_tokens": 86,
                "_completion_tokens": 10,
                "_prompt_cost_usd": 1.29e-05,
                "_completion_cost_usd": 6e-06,
                "_total_cost_usd": 1.89e-05,
            }
        ]
    )
    pd.testing.assert_frame_equal(result, expected_result)

    # test failure

    runtime.api_key = "fake_api_key"

    result = asyncio.run(
        runtime.batch_to_batch(
            batch,
            input_template="My name is {input_name} and I am {input_age} years old.",
            instructions_template="",
            response_model=Output,
        )
    )

    # note sync and async error handling behave differently in instructor. After a retry, async only returns the message while sync returns the full error object
    expected_result = pd.DataFrame.from_records(
        [
            {
                "_adala_error": True,
                "_adala_message": "AuthenticationError",
                "_adala_details": "litellm.AuthenticationError: AuthenticationError: OpenAIException - Incorrect API key provided: fake_api_key. You can find your API key at https://platform.openai.com/account/api-keys.",
                "_prompt_tokens": 9,
                "_completion_tokens": 0,
                "_prompt_cost_usd": 1.35e-06,
                "_completion_cost_usd": 0.0,
                "_total_cost_usd": 1.35e-06,
            }
        ]
    )
    pd.testing.assert_frame_equal(result, expected_result)

    # TODO test batch with successes and failures, figure out how to inject a particular error into LiteLLM
