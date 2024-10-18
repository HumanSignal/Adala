import pytest
import pandas as pd
import responses
import asyncio


@responses.activate
def test_agent_with_custom_base_url():
    from adala.agents import Agent  # type: ignore

    agent_json = {
        "skills": [
            {
                "type": "ClassificationSkill",
                "name": "ClassificationResult",
                "instructions": "",
                "input_template": "Classify sentiment of the input text: {input}",
                "field_schema": {
                    "output": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    }
                },
            }
        ],
        "runtimes": {
            "default": {
                "type": "AsyncLiteLLMChatRuntime",
                "api_version": "v1",
                "max_tokens": 4096,
                "model": "openai/llama3.1",
                "temperature": 0,
                "batch_size": 100,
                "timeout": 120,
                "verbose": False,
                "base_url": "http://localhost:11434/v1/",
                "api_key": "ollama",
                "auth_token": "SECRET-TEST-TOKEN",
            }
        },
    }
    agent = Agent(**agent_json)

    df = pd.DataFrame([["I'm happy"], ["I'm sad"], ["I'm neutral"]], columns=["input"])

    results = asyncio.run(agent.arun(input=df))
    print(results)
