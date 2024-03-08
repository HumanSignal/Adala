import fastapi
import logging
import pickle
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
from typing import Generic, TypeVar, Optional, List, Dict, Any
from typing_extensions import Annotated
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from adala.agents import Agent
from tasks.process_file import process_file
from log_middleware import LogMiddleware

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
    data: Optional[ResponseData] = None
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


@app.post("/submit", response_model=Response[JobCreated])
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
        request (JobStatusRequest): The request model for getting the status of a job.

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


class JobCancelRequest(BaseModel):
    """
    Request model for cancelling a job.
    """
    job_id: str


class JobCancelResponse(BaseModel):
    """
    Response model for cancelling a job.
    """
    status: str


@app.post('/cancel')
def cancel_job(request: JobCancelRequest):
    """
    Cancel a job.

    Args:
        request (JobCancelRequest): The request model for cancelling a job.

    Returns:
        JobCancelResponse: The response model for cancelling a job.
    """
    job = process_file.AsyncResult(request.job_id)
    job.revoke()
    return Response[JobCancelResponse](data=JobCancelResponse(status='cancelled'))
