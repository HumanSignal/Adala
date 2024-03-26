import logging
import pickle
from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar
import os

import fastapi
from adala.agents import Agent
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
import uvicorn
from redis import Redis

from log_middleware import LogMiddleware
from tasks.process_file import app as celery_app
from tasks.process_file import process_file

logger = logging.getLogger(__name__)


app = fastapi.FastAPI()

# TODO: add a correct middleware policy to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
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


class Status(Enum):
    PENDING = 'Pending'
    INPROGRESS = 'InProgress'
    COMPLETED = 'Completed'
    FAILED = 'Failed'
    CANCELED = 'Canceled'


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


@app.get('/jobs/{job_id}', response_model=Response[JobStatusResponse])
def get_status(job_id):
    """
    Get the status of a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse: The response model for getting the status of a job.
    """
    celery_status_map = {
        'PENDING': Status.PENDING,
        'STARTED': Status.INPROGRESS,
        'SUCCESS': Status.COMPLETED,
        'FAILURE': Status.FAILED,
        'REVOKED': Status.CANCELED,
        'RETRY': Status.INPROGRESS
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


@app.delete('/jobs/{job_id}', response_model=Response[JobStatusResponse])
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
    return Response[JobStatusResponse](
        data=JobStatusResponse(status=Status.CANCELED)
    )


@app.get('/health')
async def health():
    """
    Check if the app is alive.

    If app is alive (e.g. started), returns status code 200.
    """
    return {'status': 'ok'}


@app.get('/ready')
async def ready():
    """
    Check if the app is ready to serve requests.

    See if we can reach redis. If not, raise a 500 error. Else, return 200.
    """
    try:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        redis_conn = Redis.from_url(redis_url, socket_connect_timeout=1)
        redis_conn.ping()
    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f'Error when checking Redis connection: {exception}',
        )

    return {'status': 'ok'}


if __name__ == '__main__':
    # for debugging
    uvicorn.run("app:app", host='0.0.0.0', port=30001)
