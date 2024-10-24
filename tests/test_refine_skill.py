import pytest
import os
from adala.agents.base import Agent
from adala.skills._base import TransformSkill
from adala.skills.collection.prompt_improvement import ImprovedPromptResponse
from unittest.mock import patch


@pytest.fixture
def agent_json():
    return {
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 200,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
        "teacher_runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_tokens": 1000,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
        "skills": [
            {
                "type": "ClassificationSkill",
                "name": "my_classification_skill",
                "instructions": "",
                "input_template": "{text} {id}",
                "field_schema": {
                    "output": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    }
                },
            }
        ],
    }


@pytest.mark.use_openai
@pytest.mark.asyncio
async def test_arefine_skill_no_input_data(client, agent_json):
    skill_name = "my_classification_skill"

    payload = {
        "agent": agent_json,
        "skill_to_improve": skill_name,
        "input_variables": ["text", "id"],
    }

    response = client.post("/improved-prompt", json=payload)

    assert response.status_code == 200
    result = response.json()

    assert "data" in result
    assert "output" in result["data"]
    output = result["data"]["output"]

    assert "reasoning" in output
    assert "new_prompt_title" in output
    assert "new_prompt_content" in output
    assert "{text}" in output["new_prompt_content"]


@pytest.mark.use_openai
@pytest.mark.asyncio
async def test_arefine_skill_with_input_data(client, agent_json):
    skill_name = "my_classification_skill"

    batch_data = [
        {"text": "This is a test text", "id": "1"},
        {"text": "This is another test text", "id": "2"},
    ]

    payload = {
        "agent": agent_json,
        "skill_to_improve": skill_name,
        "input_variables": ["text", "id"],
        "data": batch_data,
    }

    response = client.post("/improved-prompt", json=payload)

    assert response.status_code == 200
    result = response.json()

    assert "data" in result
    assert "output" in result["data"]
    output = result["data"]["output"]

    assert "reasoning" in output
    assert "new_prompt_title" in output
    assert "new_prompt_content" in output
    assert "{text}" in output["new_prompt_content"]
    assert "{id}" in output["new_prompt_content"]


@pytest.mark.use_openai
@pytest.mark.asyncio
async def test_arefine_skill_error_handling(client, agent_json):
    skill_name = "my_classification_skill"

    batch_data = None

    agent_json["teacher_runtimes"]["default"]["model"] = "nonexistent"

    payload = {
        "agent": agent_json,
        "skill_to_improve": skill_name,
        "input_variables": ["text", "id"],
        "batch_data": batch_data,
    }
    response = client.post("/improved-prompt", json=payload)
    assert response.status_code == 422

    # test runtime failure
    agent_json["teacher_runtimes"]["default"]["model"] = "gpt-4o"
    with patch("instructor.AsyncInstructor.create_with_completion") as mock_create:

        def side_effect(*args, **kwargs):
            if skill_name in str(kwargs):
                raise Exception(f"Simulated OpenAI API failure for {skill_name}")
            return mock_create.return_value

        mock_create.side_effect = side_effect

        resp = client.post(
            "/improved-prompt",
            json={
                "agent": agent_json,
                "skill_to_improve": skill_name,
                "input_variables": ["text", "id"],
            },
        )
        assert resp.raise_for_status()
        resp_json = resp.json()
        assert not resp_json["success"]
        assert (
            f"Simulated OpenAI API failure for {skill_name}"
            == resp_json["data"]["output"]["_adala_details"]
        )
