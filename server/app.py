from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
import os
import json

import fastapi
from adala.agents import Agent
from aiokafka import AIOKafkaProducer
from aiokafka.errors import UnknownTopicOrPartitionError
from fastapi import HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SerializeAsAny, field_validator
import uvicorn
from redis import Redis
import time

from server.handlers.result_handlers import ResultHandler
from server.log_middleware import LogMiddleware
from server.tasks.process_file import streaming_parent_task
from server.utils import (Settings, delete_topic, get_input_topic_name,
                          get_output_topic_name, init_logger)

logger = init_logger(__name__)


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


class SubmitStreamingRequest(BaseModel):
    """
    Request model for submitting a streaming job.
    """

    agent: Agent
    # SerializeAsAny allows for subclasses of ResultHandler
    result_handler: SerializeAsAny[ResultHandler]
    task_name: str = "streaming_parent_task"

    @field_validator("result_handler", mode="before")
    def validate_result_handler(cls, value: Dict) -> ResultHandler:
        """
        Allows polymorphism for ResultHandlers created from a dict; same implementation as the Skills, Environment, and Runtime within an Agent
        "type" is the name of the subclass of ResultHandler being used. Currently available subclasses: LSEHandler, DummyHandler
        Look in server/handlers/result_handlers.py for available subclasses
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


@app.post("/jobs/submit-streaming", response_model=Response[JobCreated])
async def submit_streaming(request: SubmitStreamingRequest):
    """
    Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.
    """

    task = streaming_parent_task
    result = task.apply_async(
        kwargs={"agent": request.agent, "result_handler": request.result_handler}
    )
    logger.info(f"Submitted {task.name} with ID {result.id}")

    return Response[JobCreated](data=JobCreated(job_id=result.id))


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

    topic = get_input_topic_name(batch.job_id)
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()

    try:
        for record in batch.data:
            await producer.send_and_wait(topic, value=record)
            # FIXME Temporary workaround for messages getting dropped.
            #       Remove once our kafka messaging is more reliable.
            time.sleep(0.1)
        logger.info(
            f"The number of records sent to input_topic:{topic} record_no:{len(batch.data)}"
        )
    except UnknownTopicOrPartitionError:
        await producer.stop()
        raise HTTPException(
            status_code=500, detail=f"{topic=} for job {batch.job_id} not found"
        )
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
    job = streaming_parent_task.AsyncResult(job_id)
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
    job = streaming_parent_task.AsyncResult(job_id)
    # try using wait=True? then what kind of timeout is acceptable? currently we don't know if we've failed to cancel a job, this always returns success
    # should use SIGTERM or SIGINT in theory, but there is some unhandled kafka cleanup that causes the celery worker to report a bunch of errors on those, will fix in a later PR
    job.revoke(terminate=True, signal="SIGKILL")

    # Delete Kafka topics
    # TODO check this doesn't conflict with parent_job_error_handler
    input_topic_name = get_input_topic_name(job_id)
    output_topic_name = get_output_topic_name(job_id)
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)

    return Response[JobStatusResponse](data=JobStatusResponse(status=Status.CANCELED))


@app.get("/health")
async def health():
    """
    Check if the app is alive.

    If app is alive (e.g. started), returns status code 200.
    """
    return {"status": "ok"}


async def _get_redis_conn():
    """
    needs to be in a separate function to allow dependency injection for testing
    """
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    redis_conn = Redis.from_url(redis_url, socket_connect_timeout=1)
    return redis_conn


@app.get("/ready")
async def ready(redis_conn: Redis = Depends(_get_redis_conn)):
    """
    Check if the app is ready to serve requests.

    See if we can reach redis. If not, raise a 500 error. Else, return 200.
    """
    try:
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
