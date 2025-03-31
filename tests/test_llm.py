import pytest
import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from adala.runtimes import (
    LiteLLMChatRuntime,
    AsyncLiteLLMChatRuntime,
    AsyncLiteLLMVisionRuntime,
)
from adala.utils.parse import MessageChunkType


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

    # test structured success with extra unused variables

    class Output(BaseModel):
        name: str = Field(..., description="name:")
        age: str = Field(..., description="age:")

    result = runtime.record_to_record(
        record={"input_name": "Carla", "input_age": 25},
        input_template="My name is {input_name} and I am {input_age:02d} years old with {brackets:.2f} and {brackets2:invalid_format_spec} and {input_name:invalid_format_spec}.",
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
    assert result["name"] == expected_result["name"]
    assert result["age"] == expected_result["age"]
    assert isinstance(result["_prompt_tokens"], int)
    assert isinstance(result["_completion_tokens"], int)
    assert isinstance(result["_prompt_cost_usd"], float)
    assert isinstance(result["_completion_cost_usd"], float)
    assert isinstance(result["_total_cost_usd"], float)

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
        "_prompt_tokens": 0,
        "_completion_tokens": 0,
        "_prompt_cost_usd": 0.0,
        "_completion_cost_usd": 0.0,
        "_total_cost_usd": 0.0,
        "_message_counts": {"text": 1},
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
                "_message_counts": {"text": 1},
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

    expected_result = pd.DataFrame.from_records(
        [
            {
                "_adala_error": True,
                "_adala_message": "AuthenticationError",
                "_adala_details": "litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: fake_api_key. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
                "_prompt_tokens": 0,
                "_completion_tokens": 0,
                "_prompt_cost_usd": 0.0,
                "_completion_cost_usd": 0.0,
                "_total_cost_usd": 0.0,
                "_message_counts": {"text": 1},
            }
        ]
    )
    pd.testing.assert_frame_equal(result, expected_result)

    # TODO test batch with successes and failures, figure out how to inject a particular error into LiteLLM


@pytest.mark.vcr
def test_vision_runtime():

    # test success

    runtime = AsyncLiteLLMVisionRuntime()

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
            }
        ]
    )

    pd.testing.assert_frame_equal(result[["name", "age"]], expected_result)

    # assert all other columns (costs) are nonzero
    assert (
        (
            result[
                [
                    "_prompt_tokens",
                    "_completion_tokens",
                    "_prompt_cost_usd",
                    "_completion_cost_usd",
                    "_total_cost_usd",
                ]
            ]
            > 0
        )
        .all()
        .all()
    )

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

    expected_result = pd.DataFrame.from_records(
        [
            {
                "_adala_error": True,
                "_adala_message": "AuthenticationError",
                "_adala_details": "litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: fake_api_key. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
            }
        ]
    )
    pd.testing.assert_frame_equal(
        result[["_adala_error", "_adala_message", "_adala_details"]], expected_result
    )
    # assert only prompt costs are zero
    assert (
        (result[["_prompt_tokens", "_prompt_cost_usd", "_total_cost_usd"]] == 0.0)
        .all()
        .all()
    )
    assert (result[["_completion_tokens", "_completion_cost_usd"]] == 0.0).all().all()

    # test with image input

    runtime = AsyncLiteLLMVisionRuntime(model="gpt-4o-mini")

    batch = pd.DataFrame.from_records(
        [
            {
                "text": "What's in this image?",
                "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/687px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
            }
        ]
    )

    class VisionOutput(BaseModel):
        description: str = Field(..., description="Description of the image")

    result = asyncio.run(
        runtime.batch_to_batch(
            batch,
            input_template="{text} {image}",
            instructions_template="Describe what you see in the image.",
            response_model=VisionOutput,
            input_field_types={
                "text": MessageChunkType.TEXT,
                "image": MessageChunkType.IMAGE_URL,
            },
        )
    )

    assert "mona lisa" in result["description"].iloc[0].lower()
    assert (
        (
            result[
                [
                    "_prompt_tokens",
                    "_completion_tokens",
                    "_prompt_cost_usd",
                    "_completion_cost_usd",
                    "_total_cost_usd",
                ]
            ]
            > 0
        )
        .all()
        .all()
    )


@pytest.mark.asyncio
async def test_arun_instructor_with_messages_exception_handling():
    """Test exception handling in arun_instructor_with_messages when ConstrainedGenerationError is raised."""
    from adala.utils.llm_utils import arun_instructor_with_messages
    from adala.utils.exceptions import ConstrainedGenerationError
    from instructor.exceptions import InstructorRetryException
    from litellm.types.utils import Usage
    from pydantic import BaseModel, Field
    from tenacity import AsyncRetrying, stop_after_attempt
    from unittest.mock import AsyncMock, patch, MagicMock

    # Define a sample response model
    class TestModel(BaseModel):
        result: str = Field(..., description="A test result")

    # Create test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    # Create a mock AsyncInstructor
    mock_client = AsyncMock()

    # Configure the mock to raise ConstrainedGenerationError
    # In instructor, max_retries is an integer, not a Retrying object
    # So we need to simulate the retry behavior here
    side_effects = [ConstrainedGenerationError()] * 3  # Will be raised 3 times
    mock_client.chat.completions.create_with_completion = AsyncMock(
        side_effect=side_effects
    )

    # Configure retry policy (3 attempts)
    # Note: In actual code, this is converted to an integer
    retries = AsyncRetrying(stop=stop_after_attempt(3))

    # Execute the function with the mocked client - need to patch at the module level
    with patch(
        "adala.utils.llm_utils.token_counter", return_value=39
    ) as mock_token_counter:
        result = await arun_instructor_with_messages(
            client=mock_client,
            messages=messages,
            response_model=TestModel,
            model="gpt-4",
            temperature=0,
            max_tokens=100,
            retries=retries,
        )

        # Verify the mock was called
        assert mock_token_counter.call_count > 0

    # Verify the error response
    assert result["_adala_error"] is True
    assert result["_adala_message"] == "ConstrainedGenerationError"
    assert (
        "could not generate a properly-formatted response" in result["_adala_details"]
    )

    # Verify the usage statistics - using actual values from the log output
    assert result["_prompt_tokens"] == 117  # Actual token count from execution
    assert result["_completion_tokens"] == 0
    assert result["_total_cost_usd"] == 0.00351
    assert result["_prompt_cost_usd"] == 0.00351
    assert result["_completion_cost_usd"] == 0.0

    # Verify the mock was called the expected number of times
    # The actual retries happen inside instructor, not in our code
    assert mock_client.chat.completions.create_with_completion.call_count == 1
