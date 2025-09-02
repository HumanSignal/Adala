import time
import httpx
import os
import asyncio
import pytest
import base64
import json
from tempfile import NamedTemporaryFile
import pandas as pd
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from fastapi.testclient import TestClient

# TODO manage which keys correspond to which models/deployments, probably using a litellm Router
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
LS_API_KEY = os.getenv("LS_API_KEY")

SUBMIT_PAYLOAD = {
    "agent": {
        "environment": {
            "type": "AsyncKafkaEnvironment",
            "kafka_kwargs": {"kafka_bootstrap_servers": ""},
            "kafka_input_topic": "",
            "kafka_output_topic": "",
            "timeout_ms": 1,
        },
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
                "max_tokens": 200,
                "temperature": 0,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
    },
    "result_handler": {"type": "DummyHandler"},
}


async def arun_job(
    client: httpx.AsyncClient,
    streaming_payload: dict,
    batch_payload_datas: list[list[dict]],
):
    streaming_response = await client.post(
        "/jobs/submit-streaming", json=streaming_payload
    )
    streaming_response.raise_for_status()

    job_id = streaming_response.json()["data"]["job_id"]

    batch_responses = []
    for batch_payload_data in batch_payload_datas:
        await asyncio.sleep(1)
        batch_payload = {
            "job_id": job_id,
            "data": batch_payload_data,
        }
        resp = await client.post("/jobs/submit-batch", json=batch_payload)
        resp.raise_for_status()
        batch_responses.append(resp)

    return streaming_response, batch_responses


async def await_for_terminal_status(client, job_id, timeout_sec, poll_interval_sec):
    terminal_statuses = ["Completed", "Failed", "Canceled"]
    for _ in range(int(timeout_sec / poll_interval_sec)):
        resp = await client.get(f"/jobs/{job_id}")
        status = resp.json()["data"]["status"]
        if status in terminal_statuses:
            break
        print(f"polling {job_id=} ", status, flush=True)
        await asyncio.sleep(poll_interval_sec)
    return status


async def arun_job_and_get_output(
    client: httpx.AsyncClient,
    streaming_payload_agent: dict,
    batch_payload_datas: list[list[dict]],
    timeout_sec=10,
    poll_interval_sec=1,
) -> pd.DataFrame:
    with NamedTemporaryFile(mode="r") as f:
        streaming_payload = {
            "agent": streaming_payload_agent,
            "result_handler": {"type": "CSVHandler", "output_path": f.name},
        }

        resp, _ = await arun_job(client, streaming_payload, batch_payload_datas)
        job_id = resp.json()["data"]["job_id"]

        status = await await_for_terminal_status(
            client, job_id, timeout_sec, poll_interval_sec
        )
        assert status == "Completed", status

        output = pd.read_csv(f.name).set_index("task_id")

    return output


async def arun_n_jobs(n: int):
    async with httpx.AsyncClient(
        timeout=10, base_url="http://localhost:30001"
    ) as client:
        response_groups = await asyncio.gather(*[arun_job(client) for _ in range(n)])
    return response_groups


def test_health_endpoint(client):
    resp = client.get("/health")

    result = resp.json()["status"]
    assert result == "ok", f"Expected status = ok, but instead returned {result}."


def test_ready_endpoint(client, redis_mock):
    resp = client.get("/ready")

    result = resp.json()["status"]
    assert result == "ok", f"Expected status = ok, but instead returned {result}."


@pytest.mark.parametrize(
    "input_data, skills, output_column",
    [
        # text classification
        (
            [
                {"task_id": 1, "text": "anytexthere", "output": "Feature Lack"},
                {"task_id": 2, "text": "othertexthere", "output": "Feature Lack"},
                {"task_id": 3, "text": "anytexthere", "output": "Feature Lack"},
                {"task_id": 4, "text": "othertexthere", "output": "Feature Lack"},
            ],
            [
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
            "output",
        ),
        # entity extraction
        (
            [
                {
                    "task_id": 1,
                    "text": "John Doe, 26 years old, works at Google",
                    "entities": [
                        {"start": 0, "end": 8, "label": "PERSON"},
                        {"start": 26, "end": 36, "label": "AGE"},
                        {"start": 47, "end": 53, "label": "ORG"},
                    ],
                },
                {
                    "task_id": 2,
                    "text": "Jane Doe, 30 years old, works at Microsoft",
                    "entities": [
                        {"start": 0, "end": 8, "label": "PERSON"},
                        {"start": 26, "end": 36, "label": "AGE"},
                        {"start": 47, "end": 55, "label": "ORG"},
                    ],
                },
                {
                    "task_id": 3,
                    "text": "John Smith, 40 years old, works at Amazon",
                    "entities": [
                        {"start": 0, "end": 10, "label": "PERSON"},
                        {"start": 28, "end": 38, "label": "AGE"},
                        {"start": 49, "end": 55, "label": "ORG"},
                    ],
                },
                {
                    "task_id": 4,
                    "text": "Jane Smith, 35 years old, works at Facebook",
                    "entities": [
                        {"start": 0, "end": 10, "label": "PERSON"},
                        {"start": 28, "end": 38, "label": "AGE"},
                        {"start": 49, "end": 57, "label": "ORG"},
                    ],
                },
            ],
            [
                {
                    "type": "EntityExtraction",
                    "name": "entity_extraction",
                    "input_template": 'Extract entities from the input text.\n\nInput:\n"""\n{text}\n"""',
                    "labels": ["PERSON", "AGE", "ORG"],
                }
            ],
            "entities",
        ),
    ],
)
@pytest.mark.use_openai
@pytest.mark.use_server
def test_streaming_use_cases(client, input_data, skills, output_column):
    data = pd.DataFrame.from_records(input_data)
    batch_data = data.drop(output_column, axis=1).to_dict(orient="records")
    submit_payload = deepcopy(SUBMIT_PAYLOAD)

    with NamedTemporaryFile(mode="r") as f:

        submit_payload["agent"]["skills"] = skills

        submit_payload["result_handler"] = {
            "type": "CSVHandler",
            "output_path": f.name,
        }

        resp = client.post("/jobs/submit-streaming", json=submit_payload)
        resp.raise_for_status()
        job_id = resp.json()["data"]["job_id"]

        batch_payload = {
            "job_id": job_id,
            "data": batch_data[:2],
        }
        resp = client.post("/jobs/submit-batch", json=batch_payload)
        resp.raise_for_status()
        time.sleep(1)
        batch_payload = {
            "job_id": job_id,
            "data": batch_data[2:],
        }
        resp = client.post("/jobs/submit-batch", json=batch_payload)
        resp.raise_for_status()

        timeout_sec = 10
        poll_interval_sec = 1
        terminal_statuses = ["Completed", "Failed", "Canceled"]
        for _ in range(int(timeout_sec / poll_interval_sec)):
            resp = client.get(f"/jobs/{job_id}")
            status = resp.json()["data"]["status"]
            if status in terminal_statuses:
                break
            print("polling ", status)
            time.sleep(poll_interval_sec)
        assert status == "Completed", status

        output = pd.read_csv(f.name).set_index("task_id")
        assert not output["error"].any(), "adala returned errors"

        # check for expected output
        expected_outputs = data.set_index("task_id")[output_column].tolist()
        actual_outputs = [eval(item)[output_column] for item in output.output.tolist()]
        for actual_output, expected_output in zip(actual_outputs, expected_outputs):
            if skills[0]["type"] == "EntityExtraction":
                # Live generations may be flaky, check only 3 entities are presented
                actual_labels = [entity["label"] for entity in actual_output]
                expected_labels = [entity["label"] for entity in expected_output]
                assert actual_labels == expected_labels
                continue

            assert (
                actual_output == expected_output
            ), "adala did not return expected output"


@pytest.mark.use_openai
@pytest.mark.use_server
@pytest.mark.asyncio
async def test_streaming_n_concurrent_requests(async_client):
    client = async_client

    # TODO test with n_requests > number of celery workers
    n_requests = 3

    data = pd.DataFrame.from_records(
        [
            {"task_id": 1, "text": "anytexthere", "output": "Feature Lack"},
            {"task_id": 2, "text": "othertexthere", "output": "Feature Lack"},
            {"task_id": 3, "text": "anytexthere", "output": "Feature Lack"},
            {"task_id": 4, "text": "othertexthere", "output": "Feature Lack"},
        ]
    )
    batch_payload_data = data.drop("output", axis=1).to_dict(orient="records")
    batch_payload_datas = [batch_payload_data[:2], batch_payload_data[2:]]

    # this sometimes takes too long and flakes, set timeout_sec if behavior continues
    outputs = await asyncio.gather(
        *[
            arun_job_and_get_output(
                client, SUBMIT_PAYLOAD["agent"], batch_payload_datas
            )
            for _ in range(n_requests)
        ]
    )

    for output in outputs:
        assert not output["error"].any(), "adala returned errors"
        expected_outputs = data.set_index("task_id")["output"].tolist()
        actual_outputs = [eval(item)["output"] for item in output.output.tolist()]
        for actual_output, expected_output in zip(actual_outputs, expected_outputs):
            assert (
                actual_output == expected_output
            ), "adala did not return expected output"


@pytest.mark.skip(
    reason='TODO: @matt-bernstein Failed at assert status == "Failed", probably existed before the skip'
)
@pytest.mark.use_openai
@pytest.mark.use_server
@pytest.mark.asyncio
async def test_streaming_submit_edge_cases(client, async_client):
    job_id = "nonexistent"

    # get status
    # TODO should fail, but that's a long way away
    resp = client.get(f"/jobs/{job_id}")
    resp.raise_for_status()
    assert resp.json()["data"]["status"] == "Pending"

    # send a full batch
    # TODO should return 400?
    batch_payload = {
        "job_id": job_id,
        "data": [
            {"task_id": 1, "text": "anytexthere"},
        ],
    }
    resp = client.post("/jobs/submit-batch", json=batch_payload)
    assert resp.status_code == 500
    assert resp.json() == {
        "detail": f"topic='adala-input-{job_id}' for job {job_id} not found"
    }

    # start a job
    resp = client.post("/jobs/submit-streaming", json=SUBMIT_PAYLOAD)
    resp.raise_for_status()
    job_id = resp.json()["data"]["job_id"]

    # send a full batch
    batch_payload = {
        "job_id": job_id,
        "data": [
            {"task_id": 1, "text": "anytexthere"},
        ],
    }
    resp = client.post("/jobs/submit-batch", json=batch_payload)
    resp.raise_for_status()

    # send a batch with no task id
    # TODO will fail in result handler, but should that make this request fail?
    batch_payload = {
        "job_id": job_id,
        "data": [
            {"text": "anytexthere"},
        ],
    }
    resp = client.post("/jobs/submit-batch", json=batch_payload)
    resp.raise_for_status()

    # send a batch with invalid task id
    # TODO will fail in result handler, but should that make this request fail?
    batch_payload = {
        "job_id": job_id,
        "data": [
            {"task_id": "not a task id", "text": "anytexthere"},
        ],
    }
    resp = client.post("/jobs/submit-batch", json=batch_payload)
    resp.raise_for_status()

    status = await await_for_terminal_status(
        async_client, job_id, timeout_sec=10, poll_interval_sec=1
    )
    assert status == "Failed", status

    # TODO test max batch size (1MB)
    # TODO test max number of records in batch
    # TODO test sending lots of batches at once
    # TODO test startup race condition for simultaneous submit-streaming and submit-batch


@pytest.mark.use_azure
@pytest.mark.use_server
def test_streaming_azure(client):
    data = pd.DataFrame.from_records(
        [
            {"task_id": 1, "text": "anytexthere", "output": "Feature Lack"},
            {"task_id": 2, "text": "othertexthere", "output": "Feature Lack"},
            {"task_id": 3, "text": "anytexthere", "output": "Feature Lack"},
            {"task_id": 4, "text": "othertexthere", "output": "Feature Lack"},
        ]
    )
    batch_data = data.drop("output", axis=1).to_dict(orient="records")

    with NamedTemporaryFile(mode="r") as f:

        SUBMIT_PAYLOAD["agent"]["default"] = {
            "type": "AsyncLiteLLMChatRuntime",
            "model": "azure/gpt35turbo",
            "api_key": AZURE_API_KEY,
            "base_url": "https://humansignal-openai-test.openai.azure.com/",
            "api_version": "2024-06-01",
            "max_tokens": 10,
            "temperature": 0,
            "batch_size": 100,
            "timeout": 10,
            "verbose": False,
        }

        SUBMIT_PAYLOAD["result_handler"] = {
            "type": "CSVHandler",
            "output_path": f.name,
        }

        resp = client.post("/jobs/submit-streaming", json=SUBMIT_PAYLOAD)
        resp.raise_for_status()
        job_id = resp.json()["data"]["job_id"]

        batch_payload = {
            "job_id": job_id,
            "data": batch_data[:2],
        }
        resp = client.post("/jobs/submit-batch", json=batch_payload)
        resp.raise_for_status()
        time.sleep(1)
        batch_payload = {
            "job_id": job_id,
            "data": batch_data[2:],
        }
        resp = client.post("/jobs/submit-batch", json=batch_payload)
        resp.raise_for_status()

        timeout_sec = 10
        poll_interval_sec = 1
        terminal_statuses = ["Completed", "Failed", "Canceled"]
        for _ in range(int(timeout_sec / poll_interval_sec)):
            resp = client.get(f"/jobs/{job_id}")
            status = resp.json()["data"]["status"]
            if status in terminal_statuses:
                break
            print("polling ", status)
            time.sleep(poll_interval_sec)
        assert status == "Completed", status

        output = pd.read_csv(f.name).set_index("task_id")
        assert not output["error"].any(), "adala returned errors"
        expected_outputs = data.set_index("task_id")["output"].tolist()
        actual_outputs = [eval(item)["output"] for item in output.output.tolist()]
        for actual_output, expected_output in zip(actual_outputs, expected_outputs):
            assert (
                actual_output == expected_output
            ), "adala did not return expected output"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_estimate_cost_endpoint_success(async_client):
    """Test the estimate-cost endpoint with a valid payload."""
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
                    "api_key": "123",
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
        "provider": "OpenAI",
    }

    resp = await async_client.post("/estimate-cost", json=req)
    assert resp.status_code == 200

    resp_data = resp.json()["data"]
    assert "prompt_cost_usd" in resp_data
    assert "completion_cost_usd" in resp_data
    assert "total_cost_usd" in resp_data
    assert "is_error" in resp_data

    assert resp_data["is_error"] is False
    assert isinstance(resp_data["prompt_cost_usd"], float)
    assert isinstance(resp_data["completion_cost_usd"], float)
    assert isinstance(resp_data["total_cost_usd"], float)
    assert np.isclose(
        resp_data["total_cost_usd"],
        resp_data["prompt_cost_usd"] + resp_data["completion_cost_usd"],
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_estimate_cost_endpoint_invalid_model(async_client):
    """Test the estimate-cost endpoint with an invalid model."""
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
                    "model": "nonexistent-model",
                    "api_key": "fake_api_key",
                    "provider": "custom",
                }
            },
        },
        "prompt": "test {text}",
        "substitutions": [{"text": "test"}],
        "provider": "Custom",
    }

    resp = await async_client.post("/estimate-cost", json=req)
    assert resp.status_code == 200

    resp_data = resp.json()["data"]
    assert "is_error" in resp_data
    assert resp_data["is_error"] is True
    assert resp_data["error_type"] is not None
    assert "not found" in resp_data["error_message"]


@pytest.mark.vcr
def test_chat_completion_endpoint_api_client(client):
    """Test the chat/completion endpoint using sync requests with OpenAI format validation."""

    # Make a direct request using the existing test client fixture
    encoded_credentials = base64.b64encode(
        json.dumps(
            {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "iat": time.time(),
                "iss": "test",
                "exp": time.time() + 3600,
                "sub": "test",
                "additional": "payload",
            }
        ).encode("utf-8")
    ).decode("utf-8")
    response = client.post(
        "/chat/completions",
        headers={
            "Authorization": f"Bearer {encoded_credentials}",
        },
        json={
            "model": "gemini/gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )

    assert response.status_code == 200
    response_data = response.json()

    # Verify the response structure matches OpenAI's format
    assert "choices" in response_data
    assert "model" in response_data
    assert "usage" in response_data
    assert len(response_data["choices"]) > 0
    assert response_data["choices"][0]["message"]["content"] is not None

    # Validate the response data object structure and content
    assert response_data["id"] is not None
    assert response_data["object"] == "chat.completion"
    assert isinstance(response_data["created"], int)
    assert response_data["model"] is not None

    # Validate choices structure
    choices = response_data["choices"]
    assert len(choices) > 0
    choice = choices[0]
    assert "finish_reason" in choice
    assert "index" in choice
    assert "message" in choice

    # Validate message content
    message = choice["message"]
    assert "content" in message
    assert message["content"] is not None
    assert "role" in message
    assert message["role"] == "assistant"

    # Validate the actual response content contains expected text
    content = message["content"]
    assert "thank you" in content.lower()

    # Validate usage structure
    usage = response_data["usage"]
    assert "completion_tokens" in usage
    assert "prompt_tokens" in usage
    assert "total_tokens" in usage
    assert isinstance(usage["completion_tokens"], int)
    assert isinstance(usage["prompt_tokens"], int)
    assert isinstance(usage["total_tokens"], int)
    assert usage["completion_tokens"] > 0
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] > 0

    # Validate service_tier exists
    assert "service_tier" in response_data


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_completion_endpoint_openai_client():
    """Test the chat/completion endpoint using AsyncOpenAI client with mocked test server."""
    from server.app import app

    # Create async httpx client with ASGITransport for mocked test server
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
    )

    # Create AsyncOpenAI client with the mocked httpx client
    client = AsyncOpenAI(
        base_url="http://localhost:30001",
        api_key=os.getenv("GEMINI_API_KEY"),
        http_client=http_client,
    )

    response = await client.chat.completions.create(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

    # Verify the response structure matches OpenAI's format
    assert hasattr(response, "choices")
    assert hasattr(response, "model")
    assert hasattr(response, "usage")
    assert len(response.choices) > 0
    assert "thank you" in response.choices[0].message.content.lower()

    # Clean up
    await http_client.aclose()


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_chat_completion_endpoint_error_handling():
    """Test error handling when using non-existent endpoint via extra_headers."""
    from server.app import app

    # Create async httpx client with ASGITransport for mocked test server
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:30001"
    )

    # note that here we use a different schema to pass the `api_key`.
    # the reason is because different providers may require additional credentials.
    encoded_credentials = base64.b64encode(
        json.dumps({"api_key": os.getenv("GEMINI_API_KEY")}).encode("utf-8")
    ).decode("utf-8")
    client = AsyncOpenAI(
        base_url="http://localhost:30001",
        api_key=encoded_credentials,
        http_client=http_client,
    )

    # First test with valid configuration to ensure it passes
    response = await client.chat.completions.create(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

    # Verify the response structure matches OpenAI's format
    assert hasattr(response, "choices")
    assert hasattr(response, "model")
    assert hasattr(response, "usage")
    assert len(response.choices) > 0
    assert "thank you" in response.choices[0].message.content.lower()

    # Test should raise an exception due to non-existent endpoint in extra_headers
    with pytest.raises(Exception):
        response = await client.chat.completions.create(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            # This will raise an exception as the endpoint does not exist
            extra_headers={"base_url": "http://non.existent.endpoint"},
        )

    # Clean up
    await http_client.aclose()


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_custom_provider_with_anthropic_endpoint():
    """Test using OpenAI client with `Custom` provider configuration pointing to Anthropic API."""
    from server.app import app

    # Create async httpx client with ASGITransport for mocked test server
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))

    # Configure credentials for custom provider pointing to Anthropic
    encoded_credentials = base64.b64encode(
        json.dumps(
            {
                "api_key": os.getenv("ANTHROPIC_API_KEY", "test-key"),
                "provider": "Custom",
                "base_url": "https://api.anthropic.com/v1/",
            }
        ).encode("utf-8")
    ).decode("utf-8")

    client = AsyncOpenAI(
        base_url="http://localhost:30001",
        api_key=encoded_credentials,
        http_client=http_client,
    )

    response = await client.chat.completions.create(
        model="claude-opus-4-1-20250805",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )

    # Verify the response structure matches OpenAI's format
    assert hasattr(response, "choices")
    assert hasattr(response, "model")
    assert hasattr(response, "usage")
    assert len(response.choices) > 0
    assert "thank you" in response.choices[0].message.content.lower()

    # Test should handle the invalid model name
    with pytest.raises(Exception) as exc_info:
        response = await client.chat.completions.create(
            model="invalid-model-name",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

    # Verify that the exception is related to API format or authentication issues
    # rather than configuration problems
    assert exc_info.value is not None

    # Clean up
    await http_client.aclose()
