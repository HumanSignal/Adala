import fastapi
from fastapi.middleware.cors import CORSMiddleware
from typing import Generic, TypeVar, Optional, List, Dict
from pydantic import BaseModel
from adala.server.tasks import process_file


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


class SubmitRequestInput(BaseModel):
    uri: str
    pass_through_columns: Optional[List] = None


class SubmitRequestOutput(BaseModel):
    result_uri: str
    error_uri: Optional[str] = None


class SubmitRequestModel(BaseModel):
    type: str
    provider: str
    provider_model: str
    api_key: str


class SubmitRequestPrompt(BaseModel):
    instructions: str
    input_template: str
    output_template: str
    constraints: List[str]


class SubmitRequest(BaseModel):
    input: SubmitRequestInput
    output: SubmitRequestOutput
    model: SubmitRequestModel
    prompt: SubmitRequestPrompt


@app.get("/")
def get_index():
    return {"status": "ok"}


@app.post("/submit", response_model=Response[JobCreated])
async def submit(request: SubmitRequest):

    # TODO: @pakelley should we adopt the schema in the request or leave the conversion here?
    serialized_agent = [{
        'skills': [{
            'name': 'text_classifier',
            'type': 'classification',
            'input_template': request.prompt.input_template,
            'output_template': request.prompt.output_template,
            'instructions': request.prompt.instructions,
            'labels': request.prompt.constraints
        }],
        'runtimes': {
            'default': {
                'type': 'async-openai-chat',
                'model': request.model.provider_model,
                'api_key': request.model.api_key
            }
        }
    }]

    result = process_file.delay(
        input_file=request.input.uri,
        serialized_agent=serialized_agent,
        output_file=request.output.result_uri,
        error_file=request.output.error_uri,
        output_columns=request.input.pass_through_columns)

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
