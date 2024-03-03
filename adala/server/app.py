import time

import fastapi
import logging
import pickle
import os
import asyncio
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Request, Depends, HTTPException
from typing import Generic, TypeVar, Optional, List, Dict, Any
from typing_extensions import Annotated
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from jose import JWTError, jwt
from adala.agents import Agent
from adala.server.tasks.process_file import process_file
from adala.environments.kafka import get_data_stream
from log_middleware import LogMiddleware

logger = logging.getLogger(__name__)


app = fastapi.FastAPI()
SECRET_KEY = os.getenv("SECRET_KEY", "unsafe_default_secret_key")
ALGORITHM = "HS256"

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


def create_token(data: dict, expires_time_seconds: int = 60):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(seconds=expires_time_seconds)
    to_encode.update({"expire": expire.isoformat()})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


class Response(BaseModel, Generic[ResponseData]):
    """
    A generic response model.

    Attributes:
        success (bool): Indicates if the request was successful. Default is True.
        data (Optional[ResponseData]): The data returned by the request. Can be any type.
        message (Optional[str]): An optional message returned by the request.
        errors (Optional[list]): An optional list of errors returned by the request.
    """

    success: bool = True
    data: Optional[ResponseData] = None
    message: Optional[str] = None
    errors: Optional[list] = None

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore
        """
        Exclude `null` values from the response.

        This method overrides the BaseModel's model_dump method to exclude keys with `None` values from the response.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: The response dictionary without keys with `None` values.
        """
        kwargs.pop("exclude_none", None)
        return super().model_dump(*args, exclude_none=True, **kwargs)


class JobCreated(BaseModel):
    """
    Response model for a job created.

    Attributes:
        job_id (str): The job id.
        token (str): The token for the job. This token is used to validate the job id.
    """
    job_id: str
    token: str = Annotated[str, AfterValidator(lambda x: len(x) > 0)]


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
    # TODO: pickle serialization doesn't work with OpenAI client, better to move it in __getstate__() but that also doesn't work
    request.agent.runtimes['default']._client = None

    serialized_agent = pickle.dumps(request.agent)
    logger.debug(f"Submitting task {task.name} with agent {serialized_agent}")
    result = task.delay(serialized_agent=serialized_agent)
    logger.debug(f"Task {task.name} submitted with job_id {result.id}")
    token = create_token({"job_id": result.id})
    return Response[JobCreated](data=JobCreated(job_id=result.id, token=token))
    # token = create_token({"job_id": "12345"})
    # return Response[JobCreated](data=JobCreated(job_id='12345', token=token))


class JobStatusRequest(BaseModel):
    """
    Request model for getting the status of a job.
    """
    job_id: str


class JobStatusResponse(BaseModel):
    """
    Response model for getting the status of a job.

    Attributes:
        status (str): The status of the job.
        processed_total (List[int]): The total number of processed records and the total number of records in job.
            Example: [10, 100] means 10% of the completeness.
    """
    status: str
    # processed_total: List[int] = Annotated[List[int], AfterValidator(lambda x: len(x) == 2)]


@app.post('/get-status')
def get_status(request: JobStatusRequest):
    """
    Get the status of a job.

    Args:
        request (JobStatusRequest): The request model for getting the status of a job.

    Returns:
        JobStatusResponse: The response model for getting the status of a job.
    """
    job = process_file.AsyncResult(request.job_id)
    try:
        status = job.status
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        status = 'FAILURE'
    else:
        logger.info(f"Job {request.job_id} status: {status}")
    return Response[JobStatusResponse](data=JobStatusResponse(status=status))


class GetPredictionStreamRequest(BaseModel):
    """
    Using event stream, get the prediction results of a job as they are processed.

    Attributes:
        job_id (str): The job id.
        token (str): The token for the job. This JWT token to verify the job id.
        topic (str): The Kafka topic to get the prediction stream.
    """
    job_id: str
    token: str
    topic: str


def verify_token_info(job_id: str, token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        encrypted_job_id: str = payload.get("job_id")
        if encrypted_job_id != job_id:
            raise HTTPException(status_code=400, detail="Invalid token")
        # Add more checks here if necessary, e.g., token used flag, expiration, etc.
        pass
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")


async def iter_data_stream(data_stream):
    while True:
        try:
            data = await asyncio.wait_for(anext(data_stream), timeout=60.0)
            yield f"data: {data}\n\n"
        except asyncio.TimeoutError:
            logger.info("Data stream heartbeat....")
            yield ":heartbeat\n\n"
        except StopAsyncIteration as e:
            logger.info(f"Data stream ended: {e}")
            break


@app.get('/prediction-stream')
def get_prediction_stream(job_id: str, token: str, topic: str):
    """
    Using event stream, get the prediction results of a job as they are processed.

    Args:
        request (GetPredictionStreamRequest): The request model for getting the prediction stream.

    Returns:
        StreamingResponse: The prediction stream.
    """
    verify_token_info(job_id, token)
    data_stream = get_data_stream(
        input_topic=topic,
        # TODO: these parameters should not be hardcoded
        bootstrap_servers='localhost:9093',
        group_id='adala-consumer-group-output',
        value_deserializer=lambda x: x.decode('utf-8'),
        timeout=60
    )

    return StreamingResponse(iter_data_stream(data_stream), media_type="text/event-stream")


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
