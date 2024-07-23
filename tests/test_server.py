import time
import requests
import httpx
import os
import asyncio
import pytest
import openai_responses
from tempfile import NamedTemporaryFile
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
                "instructions": "",
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
                "concurrent_clients": 100,
                "batch_size": 100,
                "timeout": 10,
                "verbose": False,
            }
        },
    },
    "result_handler": {"type": "DummyHandler"},
}

SUBMIT_PAYLOAD_HUMOR = {
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
                "instructions": "",
                "input_template": "{text}",
                "output_template": "{output}",
                "labels": {
                    "output": [
                        "humor",
                        "not humor",
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
                "concurrent_clients": 100,
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
    timeout_sec=60*60*2,
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
def test_streaming_10000(multiclient):
    client = multiclient
    f = open('jsonrecords.json')
    import json 
    # returns JSON object as 
    # a dictionary
    datarecord = json.load(f)

    data = pd.DataFrame.from_records(
        datarecord
    )
    batch_data = data.drop("output", axis=1).to_dict(orient="records")
    expected_output = data.set_index("task_id")["output"]

    with NamedTemporaryFile(mode="r") as f:

        print("filename", f.name, flush=True)

        SUBMIT_PAYLOAD_HUMOR["result_handler"] = {
            "type": "CSVHandler",
            "output_path": f.name,
        }

        resp = client.post("/jobs/submit-streaming", json=SUBMIT_PAYLOAD_HUMOR)
        resp.raise_for_status()
        job_id = resp.json()["data"]["job_id"]
        batchstartidx=0
        batchendidx = 1000
        for i in range(0,10):

            batch_payload = {
                "job_id": job_id,
                "data": batch_data[batchstartidx:batchendidx],
            }
            resp = client.post("/jobs/submit-batch", json=batch_payload)
            resp.raise_for_status()
            batchendidx+=1000
            batchstartidx+=1000
        # time.sleep(1)
        # batch_payload = {
        #     "job_id": job_id,
        #     "data": batch_data[2:],
        # }
        # resp = client.post("/jobs/submit-batch", json=batch_payload)
        # resp.raise_for_status()

        timeout_sec = 60*60*3
        poll_interval_sec = 1
        terminal_statuses = ["Completed", "Failed", "Canceled"]
        for _ in range(int(timeout_sec / poll_interval_sec)):
            resp = client.get(f"/jobs/{job_id}")
            status = resp.json()["data"]["status"]
            if status in terminal_statuses:
                print("terminal polling ", status, flush=True)
                break
            print("polling ", status, flush=True)
            time.sleep(poll_interval_sec)
        assert status == "Completed", status

        output = pd.read_csv(f.name).set_index("task_id")
        print(f"dataframe length, {len(output.index)}")
        output.to_json('outputresult.json', orient='records', lines=True)
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

@pytest.mark.use_openai
@pytest.mark.use_server
@pytest.mark.asyncio
async def test_streaming_2_concurrent_requests_100000_single_producer(async_client):
    client = async_client

    # TODO test with n_requests > number of celery workers
    n_requests = 2

    f = open('jsonrecords.json')
    import json 
    # returns JSON object as 
    # a dictionary
    datarecord = json.load(f)

    data = pd.DataFrame.from_records(
        datarecord
    )
    batch_payload_data = data.drop("output", axis=1).to_dict(orient="records")
    batch_payload_datas = [batch_payload_data[:1000], batch_payload_data[1000:2000],batch_payload_data[2000:3000],batch_payload_data[3000:4000],batch_payload_data[4000:5000],batch_payload_data[5000:6000],batch_payload_data[6000:7000],batch_payload_data[7000:8000],batch_payload_data[8000:9000],batch_payload_data[9000:10000]]
    expected_output = data.set_index("task_id")["output"]

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


@pytest.mark.use_openai
@pytest.mark.use_server
@pytest.mark.asyncio
async def test_streaming_2_concurrent_requests_100000(multi_async_client):
    client = multi_async_client

    # TODO test with n_requests > number of celery workers
    n_requests = 2

    f = open('jsonrecords.json')
    import json 
    # returns JSON object as 
    # a dictionary
    datarecord = json.load(f)

    data = pd.DataFrame.from_records(
        datarecord
    )
    batch_payload_data = data.drop("output", axis=1).to_dict(orient="records")
    batch_payload_datas = [batch_payload_data[:1000], batch_payload_data[1000:2000],batch_payload_data[2000:3000],batch_payload_data[3000:4000],batch_payload_data[4000:5000],batch_payload_data[5000:6000],batch_payload_data[6000:7000],batch_payload_data[7000:8000],batch_payload_data[8000:9000],batch_payload_data[9000:10000]]
    expected_output = data.set_index("task_id")["output"]

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
    resp = client.post("/jobs/submit-streaming", json=payload)
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


def _build_openai_response(completion: str):
    # can extend this to handle failures, multiple completions, etc
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": completion,
                    "role": "assistant",
                },
            }
        ]
    }


@pytest.mark.asyncio
@openai_responses.mock()
async def test_streaming_openai_only(openai_mock):
    """
    Example of using openai_responses for mocking. Not possible to combine with Celery at this time.
    """

    OPENAI_API_KEY = "mocked"

    openai_mock.router.route(host="localhost").pass_through()
    openai_mock.router.route(host="127.0.0.1").pass_through()
    # # https://mharrisb1.github.io/openai-responses-python/user_guide/responses/
    openai_mock.chat.completions.create.response = _build_openai_response(
        "mocked openai chat response"
    )

    from adala.runtimes._openai import async_create_completion
    from openai import AsyncOpenAI

    openai = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
    )

    result = await async_create_completion(
        model="gpt-3.5-turbo",
        user_prompt="return the word banana",
        client=openai,
    )

    assert result == {
        "text": "mocked openai chat response",
        "_adala_error": False,
        "_adala_message": None,
        "_adala_details": None,
    }
