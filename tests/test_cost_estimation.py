#!/usr/bin/env python3
import pytest
from adala.runtimes._litellm import AsyncLiteLLMChatRuntime
from adala.runtimes.base import CostEstimate
import numpy as np
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@pytest.mark.use_openai
def test_simple_estimate_cost():
    runtime = AsyncLiteLLMChatRuntime(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    cost_estimate = runtime.get_cost_estimate(
        prompt="testing, {text}",
        substitutions=[{"text": "knock knock, who's there"}],
        output_fields=["text"],
    )

    assert isinstance(cost_estimate, CostEstimate)
    assert isinstance(cost_estimate.prompt_cost_usd, float)
    assert isinstance(cost_estimate.completion_cost_usd, float)
    assert isinstance(cost_estimate.total_cost_usd, float)
    assert np.isclose(
        cost_estimate.total_cost_usd,
        cost_estimate.prompt_cost_usd + cost_estimate.completion_cost_usd,
    )


@pytest.mark.use_openai
def test_estimate_cost_endpoint(client):
    req = {
        "agent": {
            "skills": [
                {
                    "type": "ClassificationSkill",
                    "name": "text_classifier",
                    "instructions": "Always return the answer 'Feature Lack'.",
                    "input_template": "{text}",
                    "output_template": "{output}",
                    "labels": [
                        "Feature Lack",
                        "Price",
                        "Integration Issues",
                        "Usability Concerns",
                        "Competitor Advantage",
                    ],
                }
            ],
            "runtimes": {
                "default": {
                    "type": "AsyncLiteLLMChatRuntime",
                    "model": "gpt-4o-mini",
                    "api_key": OPENAI_API_KEY,
                }
            },
        },
        "prompt": """
        test {text}

        Use the following JSON format:
        {
            "data": [
                {
                    "output": "<output>",
                    "reasoning": "<reasoning>",
                }
            ]
        }
        """,
        "substitutions": [{"text": "test"}],
    }
    resp = client.post(
        "/estimate-cost",
        json=req,
    )
    resp_data = resp.json()["data"]
    cost_estimate = CostEstimate(**resp_data)

    assert isinstance(cost_estimate, CostEstimate)
    assert isinstance(cost_estimate.prompt_cost_usd, float)
    assert isinstance(cost_estimate.completion_cost_usd, float)
    assert isinstance(cost_estimate.total_cost_usd, float)
    assert np.isclose(
        cost_estimate.total_cost_usd,
        cost_estimate.prompt_cost_usd + cost_estimate.completion_cost_usd,
    )
