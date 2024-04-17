import logging
import pickle
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union
import os
import json

import fastapi
from adala.agents import Agent
from aiokafka import AIOKafkaProducer
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SerializeAsAny, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
import uvicorn
from redis import Redis

from log_middleware import LogMiddleware
from tasks.process_file import app as celery_app
from tasks.process_file import (
    process_file,
    process_file_streaming,
    process_streaming_output,
)
from utils import get_input_topic, Settings
from server.handlers.result_handlers import ResultHandler


logger = logging.getLogger(__name__)


settings = Settings()

app = fastapi.FastAPI()

# TODO: add a correct middleware policy to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(LogMiddleware)

ResponseData = TypeVar("ResponseData")


class Response(BaseModel, Generic[ResponseData]):
    success: bool = True
    data: ResponseData
    message: Optional[str] = None
    errors: Optional[list] = None

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Exclude `null` values from the response."""
        kwargs.pop("exclude_none", None)
        return super().model_dump(*args, exclude_none=True, **kwargs)


class JobCreated(BaseModel):
    """
    Response model for a job created.
    """

    job_id: str


class BatchSubmitted(BaseModel):
    """
    Response model for a batch submitted.
    """

    job_id: str


class Status(Enum):
    PENDING = "Pending"
    INPROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELED = "Canceled"


class JobStatusResponse(BaseModel):
    """
    Response model for getting the status of a job.

    Attributes:
        status (str): The status of the job.
        processed_total (List[int]): The total number of processed records and the total number of records in job.
            Example: [10, 100] means 10% of the completeness.
    """

    status: Status
    # processed_total: List[int] = Annotated[List[int], AfterValidator(lambda x: len(x) == 2)]

    class Config:
        use_enum_values = True


class SubmitRequest(BaseModel):
    """
    Request model for submitting a job.

    Attributes:
        agent (Agent): The agent to be used for the task. Example of serialized agent:
            {
                "skills": [{
                    "type": "classification",
                    "name": "text_classifier",
                    "instructions": "Classify the text.",
                    "input_template": "Text: {text}",
                    "output_template": "Classification result: {label}",
                    "labels": {
                        "label": ['label1', 'label2', 'label3']
                    }
                }],
                "runtimes": {
                    "default": {
                        "type": "openai-chat",
                        "model": "gpt-3.5-turbo",
                        "api_key": "..."
                    }
                }
            }
        task_name (str): The name of the task to be executed by the agent.
    """

    agent: Agent
    task_name: str = "process_file"


class SubmitStreamingRequest(BaseModel):
    """
    Request model for submitting a streaming job.
    """

    agent: Agent
    # SerializeAsAny is for allowing subclasses of ResultHandler
    result_handler: SerializeAsAny[ResultHandler]
    task_name: str = "process_file_streaming"

    @field_validator("result_handler", mode="before")
    def validate_result_handler(cls, value: Dict) -> ResultHandler:
        """
        Allows polymorphism for ResultHandlers created from a dict; same implementation as the Skills, Environment, and Runtime within an Agent
        """
        if "type" not in value:
            raise HTTPException(
                status_code=400, detail="Missing type in result_handler"
            )
        result_handler = ResultHandler.create_from_registry(value.pop("type"), **value)
        return result_handler


class BatchData(BaseModel):
    """
    Model for a batch of data submitted to a streaming job
    """

    job_id: str
    data: List[dict]


@app.get("/")
def get_index():
    return {"status": "ok"}


@app.post("/jobs/submit", response_model=Response[JobCreated])
async def submit(request: SubmitRequest):
    """
    Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.
    """

    # TODO: get task by name, e.g. request.task_name
    task = process_file
    agent = request.agent

    logger.debug(f"Submitting task {task.name} with agent {agent}")
    result = task.delay(agent=agent)
    logger.debug(f"Task {task.name} submitted with job_id {result.id}")

    return Response[JobCreated](data=JobCreated(job_id=result.id))


@app.post("/jobs/submit-streaming", response_model=Response[JobCreated])
async def submit_streaming(request: SubmitStreamingRequest):
    """
    Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.
    """

    # TODO: get task by name, e.g. request.task_name
    task = process_file_streaming
    agent = request.agent

    logger.info(f"Submitting task {task.name} with agent {agent}")
    input_result = task.delay(agent=agent)
    input_job_id = input_result.id
    logger.info(f"Task {task.name} submitted with job_id {input_job_id}")

    task = process_streaming_output
    logger.info(f"Submitting task {task.name}")
    output_result = task.delay(
        job_id=input_job_id, result_handler=request.result_handler
    )
    output_job_id = output_result.id
    logger.info(f"Task {task.name} submitted with job_id {output_job_id}")

    return Response[JobCreated](data=JobCreated(job_id=input_job_id))


@app.post("/jobs/submit-batch", response_model=Response)
async def submit_batch(batch: BatchData):
    """
    Submits a batch of data to an existing streaming job.
    Will push the batch of data into Kafka in a topic specific to the job ID

    Args:
        batch (BatchData): The data to push to Kafka queue to be processed by agent.arun()

    Returns:
        Response: Generic response indicating status of request
    """

    topic = get_input_topic(batch.job_id)
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()

    try:
        for record in batch.data:
            await producer.send_and_wait(topic, value=record)
    finally:
        await producer.stop()

    return Response[BatchSubmitted](data=BatchSubmitted(job_id=batch.job_id))


@app.get("/jobs/{job_id}", response_model=Response[JobStatusResponse])
def get_status(job_id):
    """
    Get the status of a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse: The response model for getting the status of a job.
    """
    celery_status_map = {
        "PENDING": Status.PENDING,
        "STARTED": Status.INPROGRESS,
        "SUCCESS": Status.COMPLETED,
        "FAILURE": Status.FAILED,
        "REVOKED": Status.CANCELED,
        "RETRY": Status.INPROGRESS,
    }
    job = process_file.AsyncResult(job_id)
    try:
        status: Status = celery_status_map[job.status]
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        status = Status.FAILED
    else:
        logger.info(f"Job {job_id} status: {status}")
    return Response[JobStatusResponse](data=JobStatusResponse(status=status))


@app.delete("/jobs/{job_id}", response_model=Response[JobStatusResponse])
def cancel_job(job_id):
    """
    Cancel a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse[status.CANCELED]
    """
    job = process_file.AsyncResult(job_id)
    job.revoke()
    return Response[JobStatusResponse](data=JobStatusResponse(status=Status.CANCELED))


@app.get("/health")
async def health():
    """
    Check if the app is alive.

    If app is alive (e.g. started), returns status code 200.
    """
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """
    Check if the app is ready to serve requests.

    See if we can reach redis. If not, raise a 500 error. Else, return 200.
    """
    try:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        redis_conn = Redis.from_url(redis_url, socket_connect_timeout=1)
        redis_conn.ping()
    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"Error when checking Redis connection: {exception}",
        )

    return {"status": "ok"}


if __name__ == "__main__":
    # for debugging
    uvicorn.run("app:app", host="0.0.0.0", port=30001)
