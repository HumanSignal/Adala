import logging
import pickle
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union, Callable
import os
import json
import asyncio

import fastapi
from adala.agents import Agent
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
import uvicorn
from redis import Redis
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

from log_middleware import LogMiddleware
from tasks.process_file import app as celery_app
from tasks.process_file import process_file, process_file_streaming

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level='DEBUG',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class Settings(BaseSettings):
    '''
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    '''
    kafka_bootstrap_servers: Union[str, List[str]]

    model_config = SettingsConfigDict(
        env_file='.env',
    )
settings = Settings()

# this function is a no-op right now, just a demo for future work to reuse kafka topics.
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # startup events happen here - spawn reusable kafka topics/groups/partitions
    # https://stackoverflow.com/a/65183219
    # https://fastapi.tiangolo.com/advanced/events/#lifespan-function

    yield

    # shutdown events happen here

app = fastapi.FastAPI(lifespan=lifespan)

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
    Only difference from SubmitRequest is the task_name
    """

    agent: Agent
    task_name: str = "process_file_streaming"


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
    serialized_agent = pickle.dumps(request.agent)

    logger.debug(f"Submitting task {task.name} with agent {serialized_agent}")
    result = task.delay(serialized_agent=serialized_agent)
    logger.debug(f"Task {task.name} submitted with job_id {result.id}")

    return Response[JobCreated](data=JobCreated(job_id=result.id))


async def poll_for_streaming_results(job_id: str, batch_size: int, handler: Callable[List[dict], None]):
    """
    Poll for results in the kafka output topic and handle them.
    """
    print(f"Polling for results {job_id=}", flush=True)

    # have a startup time which is different from the poll interval to let the Runtime start producing results, since this function is called immediately from /submit_streaming
    startup_time_sec = 1
    poll_interval_sec = 1

    consumer = AIOKafkaConsumer(
        f"adala-output-{job_id}",
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        # group_id="adala-consumer-group",
    )
    await consumer.start()
    print(f"consumer started {job_id=}", flush=True)

    await asyncio.sleep(startup_time_sec)

    try:
        finished = False
        while not finished:
            # getmany has a timeout_ms param which in theory should be able to replace this whole manual loop, but during testing, it was blocking handling of the next request for some reason, even though it's called await internally. Could be an aiokafka bug, could be incorrect usage.
            data = await consumer.getmany(max_records=batch_size)
            print(f"got batch {data=}", flush=True)
            # should be only one topic_partition, the one we passed to AIOKafkaConsumer()
            for topic_partition, messages in data.items():
                if messages:

                    # callback/handler/connector (pick a name) executes here for each batch
                    handler(messages)

                else:
                    print(f"No messages for {job_id=}", flush=True)
                if len(messages) < batch_size:
                    # assume we're done if we get a partial batch. Not totally sold on this logic. Could try exponential backoff, etc.
                    print(f"End of stream for {job_id=}", flush=True)
                    finished = True

            await asyncio.sleep(poll_interval_sec)

    finally:
        await consumer.stop()


def poll_for_streaming_results_sync(job_id: str, batch_size: int, handler: Callable[List[dict], None]):
    # wrapper for poll_for_streaming_results to be run in a threadpool to unblock the main thread
    # this shouldn't be necessary, but there's still some problem with poll_for_streaming_results
    asyncio.run(poll_for_streaming_results(job_id, batch_size, handler))


# should probably stick all these in a separate file
def dummy_handler(batch):
    print(f"Batch: {batch}", flush=True)


@app.post("/jobs/submit-streaming", response_model=Response[JobCreated])
async def submit_streaming(request: SubmitStreamingRequest, background_tasks: fastapi.BackgroundTasks):
    """
    Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.
    """

    # TODO: get task by name, e.g. request.task_name
    task = process_file_streaming
    serialized_agent = pickle.dumps(request.agent)

    # none of these logs are printing, even with `disable_existing_loggers: False` in the uvicorn log config. No idea why. print() works at least.
    print(f"\n\n\n\n\n\nSubmitting task {task.name}", flush=True)
    logger.info(f"\n\n\n\n\n\n2Submitting task {task.name} with agent {serialized_agent}")
    logger.debug(f"\n\n\n\n\n\n3Submitting task {task.name} with agent {serialized_agent}")
    logger.critical(f"\n\n\n\n\n\n4Submitting task {task.name} with agent {serialized_agent}")

    result = task.delay(serialized_agent=serialized_agent)
    # TODO pass batch size in request?
    # TODO pass handler (and params for it) in request
    # background_tasks.add_task(poll_for_streaming_results, job_id=result.id, batch_size=2, handler=dummy_handler)
    # the invocation below should run in a threadpool instead of in the main thread, because the function is not async, but this is still blocking the main thread??
    background_tasks.add_task(poll_for_streaming_results_sync, job_id=result.id, batch_size=2, handler=dummy_handler)
    print(f"Task {task.name} submitted with job_id {result.id}")

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

    topic = f"adala-input-{batch.job_id}"
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
