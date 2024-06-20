import time
import json
import requests
import httpx
import os
import asyncio
import pytest
from fastapi.testclient import TestClient
from server.app import app
import openai_responses
from openai_responses import OpenAIMock

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LS_API_KEY = os.getenv("LS_API_KEY")

payload = {
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

if 0:
    payload["result_handler"] = {
        "type": "LSEHandler",
        "api_key": LS_API_KEY,
        "url": "http://localhost:8000",
        "modelrun_id": 184,
    }


def run_job():
    resp = requests.post("http://localhost:30001/jobs/submit-streaming", json=payload)
    job_id = resp.json()["data"]["job_id"]

    batch_payload = {
        "job_id": job_id,
        "data": [{"text": "anytexthere"}, {"text": "othertexthere"}],
    }
    resp = requests.post("http://localhost:30001/jobs/submit-batch", json=batch_payload)
    time.sleep(1)
    resp = requests.post("http://localhost:30001/jobs/submit-batch", json=batch_payload)


async def arun_job(client: httpx.AsyncClient):
    resp = await client.post(
        "http://localhost:30001/jobs/submit-streaming", json=payload
    )
    job_id = resp.json()["data"]["job_id"]

    batch_payload = {
        "job_id": job_id,
        "data": [{"text": "anytexthere"}, {"text": "othertexthere"}],
    }
    time.sleep(10)
    resp2 = await client.post(
        "http://localhost:30001/jobs/submit-batch", json=batch_payload
    )
    resp3 = await client.post(
        "http://localhost:30001/jobs/submit-batch", json=batch_payload
    )
    return resp, resp2, resp3


async def arun_n_jobs(n: int):
    async with httpx.AsyncClient(timeout=10) as client:
        response_groups = await asyncio.gather(*[arun_job(client) for _ in range(n)])
    for resps in response_groups:
        for resp in resps:
            resp.raise_for_status()
    return response_groups


# r = asyncio.run(arun_n_jobs(3))


def test_health_endpoint():
    test_client = TestClient(app)
    resp = test_client.get("/health")

    result = resp.json()["status"]
    assert result == "ok", f"Expected status = ok, but instead returned {result}."


def test_ready_endpoint(redis_mock):
    test_client = TestClient(app)
    resp = test_client.get("/ready")

    result = resp.json()["status"]
    assert result == "ok", f"Expected status = ok, but instead returned {result}."


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


# to run with the real app, just don't pass the fixtures
# def test_streaming():
# @pytest.mark.asyncio
# def test_streaming(redis_mock, celery_app_mock, celery_session_worker):
# @openai_responses.mock()
# @pytest.mark.usefixtures('celery_session_app')
# @pytest.mark.usefixtures('celery_session_worker')
@pytest.mark.skip('wip')
def test_streaming(
    # redis_mock, celery_app_mock, celery_worker, openai_mock_magic, openai_key_mock
    celery_app_mock, redis_mock, # openai_mock_magic,
):

    payload["agent"]["runtimes"]["default"]["api_key"] = os.getenv("OPENAI_API_KEY")

    # openai_mock.router.route(host="localhost").pass_through()
    # # breakpoint()
    # # https://mharrisb1.github.io/openai-responses-python/user_guide/responses/
    # openai_mock.chat.completions.create.response = {
        # "choices": [
            # {
                # "index": 0,
                # "finish_reason": "stop",
                # "message": {
                    # "content": "mocked openai chat response",
                    # "role": "assistant",
                # },
            # }
        # ]
    # }

    test_client = TestClient(app)
    resp = test_client.post("/jobs/submit-streaming", json=payload)
    job_id = resp.json()["data"]["job_id"]

    batch_payload = {
        "job_id": job_id,
        "data": [{"text": "anytexthere"}, {"text": "othertexthere"}],
    }
    resp = test_client.post("/jobs/submit-batch", json=batch_payload)
    time.sleep(1)
    resp = test_client.post("/jobs/submit-batch", json=batch_payload)



@pytest.mark.skip('wip')
@openai_responses.mock()
def test_streaming_real(openai_mock):

    openai_mock.router.route(host="localhost").pass_through()
    openai_mock.router.route(host="127.0.0.1").pass_through()
    # # breakpoint()
    # # https://mharrisb1.github.io/openai-responses-python/user_guide/responses/
    openai_mock.chat.completions.create.response = _build_openai_response(
        "mocked openai chat response"
    )

    test_client = TestClient(app)
    resp = test_client.post("/jobs/submit-streaming", json=payload)
    job_id = resp.json()["data"]["job_id"]

    batch_payload = {
        "job_id": job_id,
        "data": [{"text": "anytexthere"}, {"text": "othertexthere"}],
    }
    resp = test_client.post("/jobs/submit-batch", json=batch_payload)
    time.sleep(1)
    resp = test_client.post("/jobs/submit-batch", json=batch_payload)


@pytest.mark.asyncio
@openai_responses.mock()
async def test_streaming_celery_only(openai_mock, celery_app, celery_worker):

    # from server.tasks.process_file import app as celery_app
    from server.tasks.process_file import streaming_parent_task

    openai_mock.router.route(host="localhost").pass_through()
    openai_mock.router.route(host="127.0.0.1").pass_through()
    # # breakpoint()
    # # https://mharrisb1.github.io/openai-responses-python/user_guide/responses/
    openai_mock.chat.completions.create.response = _build_openai_response(
        "mocked openai chat response"
    )

    from adala.agents import Agent
    from server.handlers.result_handlers import ResultHandler

    agent = Agent(**payload['agent'])
    handler = ResultHandler.create_from_registry(
        payload['result_handler'].pop('type'),
        **payload['result_handler']
    )
    result = streaming_parent_task.apply_async(
        kwargs={
            'agent': agent,
            'result_handler': handler,
        }
    )
    job_id = result.id

    batch_payload = {
        "job_id": job_id,
        "data": [{"text": "anytexthere"}, {"text": "othertexthere"}],
    }

    from server.utils import get_input_topic_name
    from aiokafka import AIOKafkaProducer

    topic = get_input_topic_name(job_id)
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9093",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()

    try:
        for record in batch_payload['data']:
            await producer.send_and_wait(topic, value=record)
    finally:
        await producer.stop()

