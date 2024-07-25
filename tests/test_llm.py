import openai_responses
import pytest
import json
from pydantic import BaseModel, Field
from typing import Union, Dict


def _build_openai_response(completion: Union[str, Dict]):
    # can extend this to handle failures, multiple completions, etc
    if isinstance(completion, str):
        completion = {
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": completion,
                    "role": "assistant"
                }}]}
    return completion


class ExampleResponseModel(BaseModel):
    name: str = Field(..., description="Name of the person")
    age: int = Field(..., description="Age of the person")


@pytest.mark.parametrize(
    'response_model, chat_completion, expected_result',
    [(
        None,
        "mocked openai chat response",
        {
            "data": {
                "text": "mocked openai chat response",
            },
            # "text" is deprecated - remove it and use "data" instead
            "text": "mocked openai chat response",
            "_adala_error": False,
            "_adala_message": None,
            "_adala_details": None, }),

        (
            ExampleResponseModel,
            {
                'choices': [{
                    'finish_reason': 'stop',
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [{
                            'id': 'ai_1',
                            'type': 'function',
                            'function': {
                                'name': ExampleResponseModel.__name__,
                                'arguments': json.dumps({
                                    'name': 'Carla',
                                    'age': 25,
                                })
                            }
                        }]
                    }
                }]
            },
            {
                "data": {
                    "name": "Carla",
                    "age": 25,
                },
                'text': None,
                "_adala_error": False,
                "_adala_message": None,
                "_adala_details": None
            }
        ),
    ]
)
@openai_responses.mock()
def test_get_llm_response(openai_mock, response_model, chat_completion, expected_result):
    OPENAI_API_KEY = "mocked"

    openai_mock.router.route(host="localhost").pass_through()
    openai_mock.router.route(host="127.0.0.1").pass_through()
    # # https://mharrisb1.github.io/openai-responses-python/user_guide/responses/
    openai_mock.chat.completions.create.response = _build_openai_response(chat_completion)

    from adala.utils.llm import get_llm_response

    result = get_llm_response(
        model='gpt-3.5-turbo',
        api_key=OPENAI_API_KEY,
        user_prompt='return the word banana',
        timeout=10,
        response_model=response_model,
    )

    assert result == expected_result


@pytest.mark.parametrize(
    'response_model, chat_completion, expected_result',
    [(
        None,
        "mocked openai chat response",
        {
            "data": {
                "text": "mocked openai chat response",
            },
            # "text" is deprecated - remove it and use "data" instead
            "text": "mocked openai chat response",
            "_adala_error": False,
            "_adala_message": None,
            "_adala_details": None, }),

        (
            ExampleResponseModel,
            {
                'choices': [{
                    'finish_reason': 'stop',
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [{
                            'id': 'ai_1',
                            'type': 'function',
                            'function': {
                                'name': ExampleResponseModel.__name__,
                                'arguments': json.dumps({
                                    'name': 'Carla',
                                    'age': 25,
                                })
                            }
                        }]
                    }
                }]
            },
            {
                "data": {
                    "name": "Carla",
                    "age": 25,
                },
                'text': None,
                "_adala_error": False,
                "_adala_message": None,
                "_adala_details": None
            }
        ),
    ]
)
@pytest.mark.asyncio
@openai_responses.mock()
async def test_async_get_llm_response(openai_mock, response_model, chat_completion, expected_result):
    """
    Example of using openai_responses for mocking. Not possible to combine with Celery at this time.
    """

    OPENAI_API_KEY = "mocked"

    openai_mock.router.route(host="localhost").pass_through()
    openai_mock.router.route(host="127.0.0.1").pass_through()
    # # https://mharrisb1.github.io/openai-responses-python/user_guide/responses/
    openai_mock.chat.completions.create.response = _build_openai_response(chat_completion)

    from adala.utils.llm import async_get_llm_response

    result = await async_get_llm_response(
        model='gpt-3.5-turbo',
        api_key=OPENAI_API_KEY,
        user_prompt='return the word banana',
        timeout=10,
        response_model=response_model
    )

    assert result == expected_result
