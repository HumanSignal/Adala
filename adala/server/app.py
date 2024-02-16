import fastapi
from fastapi.middleware.cors import CORSMiddleware
from typing import Generic, TypeVar, Optional, List, Dict, Any
from typing_extensions import Annotated
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from adala.agents import Agent
from adala.server.tasks.process_file import process_file


app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

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
    job_id: str


class SubmitRequest(BaseModel):
    agent: Agent
    task_name: str = "process_file"


@app.get("/")
def get_index():
    return {"status": "ok"}


@app.post("/submit", response_model=Response[JobCreated])
async def submit(request: SubmitRequest):
    """
    Submit a request to execute task in celery.
    """
    # TODO: get task by name, e.g. request.task_name
    task = process_file

    serialized_agent = request.agent.model_dump_json()
    result = task.delay(serialized_agent=serialized_agent)
    return Response[JobCreated](data=JobCreated(job_id=result.id))


class JobStatusRequest(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: str


@app.get('/get-status')
def get_status(request: JobStatusRequest):

    job = process_file.AsyncResult(request.job_id)
    return Response[JobStatusResponse](data=JobStatusResponse(status=job.status))


class JobCancelRequest(BaseModel):
    job_id: str


class JobCancelResponse(BaseModel):
    status: str


@app.post('/cancel')
def cancel_job(request: JobCancelRequest):
    job = process_file.AsyncResult(request.job_id)
    job.revoke()
    return Response[JobCancelResponse](data=JobCancelResponse(status='cancelled'))
