import time
import requests
import httpx
import os
import asyncio
from fastapi.testclient import TestClient
from server.app import app

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
    payload['result_handler'] = {
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
