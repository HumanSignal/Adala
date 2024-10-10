#!/usr/bin/env python3
from adala.runtimes._litellm import AsyncLiteLLMChatRuntime
from adala.runtimes.base import CostEstimate
from adala.agents import Agent
from adala.skills import ClassificationSkill
import numpy as np
import os
from fastapi.testclient import TestClient
from server.app import app, CostEstimateRequest

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def test_estimate_cost_endpoint():
    test_client = TestClient(app)
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
        "prompt": "test {text}",
        "substitutions": [{"text": "test"}],
    }
    resp = test_client.post(
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
