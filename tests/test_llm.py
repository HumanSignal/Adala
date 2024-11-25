import pytest
import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from adala.runtimes import LiteLLMChatRuntime, AsyncLiteLLMChatRuntime, AsyncLiteLLMVisionRuntime
from adala.runtimes._litellm import split_message_into_chunks, MessageChunkType


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

    expected_result = pd.DataFrame.from_records(
        [
            {
                "_adala_error": True,
                "_adala_message": "AuthenticationError",
                "_adala_details": "litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': {'message': 'Incorrect API key provided: fake_api_key. You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}",
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

def test_split_message_into_chunks():
    # Test basic text-only template
    result = split_message_into_chunks(
        "Hello {name}!",
        {"name": MessageChunkType.TEXT},
        name="Alice"
    )
    assert result == [{"type": "text", "text": "Hello Alice!"}]

    # Test template with image URL
    result = split_message_into_chunks(
        "Look at this {image}",
        {"image": MessageChunkType.IMAGE_URL},
        image="http://example.com/img.jpg"
    )
    assert result == [
        {"type": "text", "text": "Look at this "},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
    ]

    # Test mixed text and image template
    result = split_message_into_chunks(
        "User {name} shared {image} yesterday",
        {
            "name": MessageChunkType.TEXT,
            "image": MessageChunkType.IMAGE_URL
        },
        name="Bob",
        image="http://example.com/photo.jpg"
    )
    assert result == [
        {"type": "text", "text": "User Bob shared "},
        {"type": "image_url", "image_url": {"url": "http://example.com/photo.jpg"}},
        {"type": "text", "text": " yesterday"}
    ]

    # Test multiple occurrences of same field
    result = split_message_into_chunks(
        "{name} is here. Hi {name}!",
        {"name": MessageChunkType.TEXT},
        name="Dave"
    )
    assert result == [{"type": "text", "text": "Dave is here. Hi Dave!"}]


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
    assert (result[["_prompt_tokens", "_completion_tokens", "_prompt_cost_usd", "_completion_cost_usd", "_total_cost_usd"]] > 0).all().all()

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
    pd.testing.assert_frame_equal(result[["_adala_error", "_adala_message", "_adala_details"]], expected_result)
    # assert only prompt costs are nonzero
    assert (result[["_prompt_tokens", "_prompt_cost_usd", "_total_cost_usd"]] > 0).all().all()
    assert (result[["_completion_tokens", "_completion_cost_usd"]] == 0).all().all()
    
    # test with image input

    runtime = AsyncLiteLLMVisionRuntime(model="gpt-4o-mini")

    batch = pd.DataFrame.from_records([{
        "text": "What's in this image?",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/687px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    }])

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
                "image": MessageChunkType.IMAGE_URL
            }
        )
    )

    assert "mona lisa" in result["description"].iloc[0].lower()
    assert (result[["_prompt_tokens", "_completion_tokens", "_prompt_cost_usd", "_completion_cost_usd", "_total_cost_usd"]] > 0).all().all()