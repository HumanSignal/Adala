#!/usr/bin/env python3
import pytest
from adala.runtimes._litellm import AsyncLiteLLMChatRuntime
from adala.runtimes.base import CostEstimate
import numpy as np
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# not requires requests to be recorded
@pytest.mark.asyncio
async def test_simple_estimate_cost_openai():
    runtime = AsyncLiteLLMChatRuntime(
        model="gpt-4o-mini", api_key=OPENAI_API_KEY, provider="openai"
    )

    cost_estimate = await runtime.get_cost_estimate_async(
        prompt="testing, {text}",
        substitutions=[{"text": "knock knock, who's there"}],
        output_fields=["text"],
        provider="openai",
    )

    assert isinstance(cost_estimate, CostEstimate)
    assert np.isclose(cost_estimate.prompt_cost_usd, 1.035e-05, rtol=1e-2, atol=1e-2)
    assert np.isclose(cost_estimate.completion_cost_usd, 2.4e-06, rtol=1e-2, atol=1e-2)
    assert np.isclose(
        cost_estimate.total_cost_usd, 1.2749999999999998e-05, rtol=1e-2, atol=1e-2
    )
    assert cost_estimate.is_error is False
    assert cost_estimate.error_type is None
    assert cost_estimate.error_message is None
    assert np.isclose(
        cost_estimate.total_cost_usd,
        cost_estimate.prompt_cost_usd + cost_estimate.completion_cost_usd,
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_simple_estimate_cost_azure():
    runtime = AsyncLiteLLMChatRuntime(
        model="azure/gpt4o",
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "123"),
        provider="azure",
        base_url="https://humansignal-openai-test.openai.azure.com/",
    )

    cost_estimate = await runtime.get_cost_estimate_async(
        prompt="testing, {text}",
        substitutions=[{"text": "knock knock, who's there"}],
        output_fields=["text"],
        provider="azure",
    )

    assert isinstance(cost_estimate, CostEstimate)
    assert np.isclose(cost_estimate.prompt_cost_usd, 0.00035, rtol=1e-2, atol=1e-2)
    assert np.isclose(cost_estimate.completion_cost_usd, 0.00006, rtol=1e-2, atol=1e-2)
    assert np.isclose(cost_estimate.total_cost_usd, 0.00041, rtol=1e-2, atol=1e-2)
    assert cost_estimate.is_error is False
    assert cost_estimate.error_type is None
    assert cost_estimate.error_message is None
    assert np.isclose(
        cost_estimate.total_cost_usd,
        cost_estimate.prompt_cost_usd + cost_estimate.completion_cost_usd,
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_simple_estimate_cost_vertex_ai():

    runtime = AsyncLiteLLMChatRuntime(
        model="vertex_ai/gemini-1.5-flash",
        vertex_credentials=os.getenv("VERTEX_CREDENTIALS", "123"),
    )

    cost_estimate = await runtime.get_cost_estimate_async(
        prompt="testing, {text}",
        substitutions=[{"text": "knock knock, who's there"}],
        output_fields=["text"],
        provider="VertexAI",
    )

    assert isinstance(cost_estimate, CostEstimate)
    assert np.isclose(cost_estimate.prompt_cost_usd, 5.25e-06, rtol=1e-2, atol=1e-2)
    assert np.isclose(cost_estimate.completion_cost_usd, 1.2e-06, rtol=1e-2, atol=1e-2)
    assert np.isclose(cost_estimate.total_cost_usd, 6.45e-06, rtol=1e-2, atol=1e-2)
    assert cost_estimate.is_error is False
    assert cost_estimate.error_type is None
    assert cost_estimate.error_message is None
    assert np.isclose(
        cost_estimate.total_cost_usd,
        cost_estimate.prompt_cost_usd + cost_estimate.completion_cost_usd,
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_simple_estimate_cost_custom_endpoint():
    runtime = AsyncLiteLLMChatRuntime(
        model="deepseek-ai/DeepSeek-V3-0324-fast",
        provider="Custom",
        base_url="https://router.huggingface.co/nebius/v1",
        api_key=os.environ.get("HF_TOKEN"),
    )

    cost_estimate = await runtime.get_cost_estimate_async(
        prompt="testing, {text}",
        substitutions=[{"text": "knock knock, who's there"}],
        output_fields=["text"],
        provider="Custom",
    )

    assert isinstance(cost_estimate, CostEstimate)
    assert cost_estimate.is_error is True
    assert cost_estimate.error_type is not None
    assert (
        "Model deepseek-ai/DeepSeek-V3-0324-fast for provider Custom not found"
        in cost_estimate.error_message
    )
    assert cost_estimate.prompt_cost_usd is None
    assert cost_estimate.completion_cost_usd is None
    assert cost_estimate.total_cost_usd is None


@pytest.mark.use_openai
async def test_estimate_cost_endpoint(async_client):
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
                    "provider": "openai",
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
    resp = await async_client.post(
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
