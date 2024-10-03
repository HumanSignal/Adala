import pytest
import os
from adala.agents.base import Agent
from adala.skills._base import TransformSkill
from adala.utils.types import ImprovedPromptResponse


@pytest.mark.asyncio
async def test_arefine_skill_no_input_data():

    agent_json = {
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
        "skills": [
            {
                "type": "ClassificationSkill",
                "name": "my_classification_skill",
                "instructions": "",
                "input_template": "Classify text: {input}",
                "field_schema": {
                    "output": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    }
                },
            }
        ],
    }
    agent = Agent(**agent_json)
    skill_name = "my_classification_skill"

    result = await agent.arefine_skill(
        skill_name,
        input_variables=["text"],
        batch_data=None,
    )

    assert isinstance(result, ImprovedPromptResponse)
    # Add more specific assertions based on expected behavior
