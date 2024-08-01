import pytest
from pydantic import BaseModel, Field
from adala.utils.llm import (
    get_llm_response,
    async_get_llm_response,
    ConstrainedLLMResponse,
    UnconstrainedLLMResponse,
)
# TODO switch these tests to use the fns from within the litellm runtimes


class ExampleResponseModel(BaseModel):
    name: str = Field(..., description="Name of the person")
    age: int = Field(..., description="Age of the person")


@pytest.mark.parametrize(
    "response_model, user_prompt, expected_result",
    [
        (
            None,
            "return the word Banana with exclamation mark",
            UnconstrainedLLMResponse(text="Banana!"),
        ),
        (
            ExampleResponseModel,
            "My name is Carla and I am 25 years old.",
            ConstrainedLLMResponse(data={"name": "Carla", "age": 25}),
        ),
    ],
)
@pytest.mark.vcr
def test_get_llm_response(response_model, user_prompt, expected_result):

    result = get_llm_response(
        user_prompt=user_prompt,
        response_model=response_model,
    )

    assert result == expected_result


@pytest.mark.parametrize(
    "response_model, user_prompt, expected_result",
    [
        (
            None,
            "return the word banana with exclamation mark",
            UnconstrainedLLMResponse(text="banana!"),
        ),
        (
            ExampleResponseModel,
            "My name is Carla and I am 25 years old.",
            ConstrainedLLMResponse(data={"name": "Carla", "age": 25}),
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_get_llm_response(response_model, user_prompt, expected_result):

    result = await async_get_llm_response(
        user_prompt=user_prompt, response_model=response_model
    )

    assert result == expected_result
