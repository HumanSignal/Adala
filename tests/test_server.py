import time
import httpx
import os
import asyncio
import pytest
from tempfile import NamedTemporaryFile
import pandas as pd


# TODO manage which keys correspond to which models/deployments, probably using a litellm Router
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
LS_API_KEY = os.getenv("LS_API_KEY")

SUBMIT_PAYLOAD = {
    "agent": {
        "environment": {
            "type": "AsyncKafkaEnvironment",
            "kafka_bootstrap_servers": "",
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
                "labels": {
                    "output": [
                        "Feature Lack",
                        "Price",
                        "Integration Issues",
                        "Usability Concerns",
                        "Competitor Advantage",
                    ]
                },
            }
        ],
        "runtimes": {
            "default": {
                "type": "AsyncOpenAIChatRuntime",
                "model": "gpt-3.5-turbo-0125",
                "api_key": OPENAI_API_KEY,
                "max_tokens": 10,
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


@pytest.mark.use_openai
@pytest.mark.use_server
def test_streaming(client):

    data = pd.DataFrame.from_records(
        [
            {"task_id": 1, "text": "anytexthere", "output": "Feature Lack"},
            {"task_id": 2, "text": "othertexthere", "output": "Feature Lack"},
            {"task_id": 3, "text": "anytexthere", "output": "Feature Lack"},
            {"task_id": 4, "text": "othertexthere", "output": "Feature Lack"},
        ]
    )
    batch_data = data.drop("output", axis=1).to_dict(orient="records")
    expected_output = data.set_index("task_id")["output"]

    with NamedTemporaryFile(mode="r") as f:

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
        assert (
            output["output"] == expected_output
        ).all(), "adala did not return expected output"


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
    expected_output = data.set_index("task_id")["output"]

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
        assert (
            output["output"] == expected_output
        ).all(), "adala did not return expected output"


@pytest.mark.skip(reason='TODO: @matt-bernstein Failed at assert status == "Failed", probably existed before the skip')
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
    expected_output = data.set_index("task_id")["output"]

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
        assert (
            output["output"] == expected_output
        ).all(), "adala did not return expected output"
