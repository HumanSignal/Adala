from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
import base64
import json
import traceback
import os
import fastapi
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from adala.agents import Agent
from adala.skills import Skill
from adala.runtimes import AsyncRuntime
from aiokafka import AIOKafkaProducer
from aiokafka.errors import UnknownTopicOrPartitionError
from fastapi import HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import litellm
from litellm.exceptions import AuthenticationError
from litellm.utils import check_valid_key, get_valid_models
from litellm import acompletion
from openai import OpenAI, AsyncOpenAI

from pydantic import BaseModel, SerializeAsAny, field_validator, Field, model_validator
from redis import Redis
import time
import uvicorn

from adala.utils.types import BatchData, ErrorResponseModel
from server.handlers.result_handlers import ResultHandler
from server.log_middleware import LogMiddleware
from adala.skills.collection.prompt_improvement import ImprovedPromptResponse
from adala.runtimes.base import CostEstimate
from server.tasks.stream_inference import streaming_parent_task
from server.utils import (
    Settings,
    delete_topic,
    get_input_topic_name,
    get_output_topic_name,
    init_logger,
)

logger = init_logger(__name__)

app = fastapi.FastAPI()

from server.worker_pool.api import router as worker_pool_router

app.include_router(worker_pool_router)

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


class BatchSubmitted(BaseModel):
    """
    Response model for a batch submitted.
    """

    job_id: str


class ModelsListRequest(BaseModel):
    provider: str


class ModelsListResponse(BaseModel):
    models_list: List[str]


class CostEstimateRequest(BaseModel):
    agent: Agent
    prompt: str
    substitutions: List[Dict]
    provider: str


class ValidateConnectionRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    vertex_credentials: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_project: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None
    endpoint: Optional[str] = None
    auth_token: Optional[str] = None
    model: Optional[str] = None


class ValidateConnectionResponse(BaseModel):
    model: str
    success: bool


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
        Look in server/handlers/result_handlers.py for available subclasses.
        """
        if "type" not in value:
            raise HTTPException(
                status_code=400, detail="Missing type in result_handler"
            )
        result_handler = ResultHandler.create_from_registry(value.pop("type"), **value)
        return result_handler


@app.get("/")
def get_index():
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Request validation error: {exc}")

    formatted_errors = []
    for error in exc.errors():
        error_type = error.get("type", "unknown_error")
        error_loc = " -> ".join(str(loc) for loc in error.get("loc", []))
        error_msg = error.get("msg", "No error message")
        formatted_errors.append(f"{error_loc}: {error_msg} ({error_type})")

    formatted_errors_str = "\n".join(formatted_errors)
    return JSONResponse(
        content=f"Validation {'errors' if len(formatted_errors) > 1 else 'error'}: {formatted_errors_str}",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


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
    settings = Settings()
    kafka_kwargs = settings.kafka.to_kafka_kwargs(client_type="producer")
    producer = AIOKafkaProducer(
        **kafka_kwargs,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",  # waits for all replicas to respond that they have written the message
    )
    await producer.start()

    try:
        for record in batch.data:
            await producer.send_and_wait(topic, value=record)

        batch_size = len(json.dumps(batch.dict()).encode("utf-8"))
        logger.info(
            f"The number of records sent to input_topic:{topic} record_no:{len(batch.data)}"
        )
        logger.info(
            f"Size of batch in bytes received for job_id:{batch.job_id} batch_size:{batch_size}"
        )
    except UnknownTopicOrPartitionError:
        raise HTTPException(
            status_code=500, detail=f"{topic=} for job {batch.job_id} not found"
        )
    finally:
        await producer.stop()

    return Response[BatchSubmitted](data=BatchSubmitted(job_id=batch.job_id))


@app.post("/validate-connection", response_model=Response[ValidateConnectionResponse])
async def validate_connection(request: ValidateConnectionRequest):
    provider = request.provider.lower()
    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    # For multi-model providers use a model that every account should have access to
    if request.model:
        if provider == "vertexai":
            model_extra = {"vertex_credentials": request.vertex_credentials}
            if request.vertex_location:
                model_extra["vertex_location"] = request.vertex_location
            if request.vertex_project:
                model_extra["vertex_project"] = request.vertex_project
        else:
            model_extra = {"api_key": request.api_key}
        try:
            response = litellm.completion(
                messages=messages,
                model=request.model,
                max_tokens=10,
                temperature=0.0,
                **model_extra,
            )
        except AuthenticationError:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model '{request.model}' is not available with your api_key / credentials",
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error validating credentials for provider {provider}: {e}",
            )

    # For single-model connections use the provided model
    else:
        if provider.lower() == "azureopenai":
            model = "azure/" + request.deployment_name
            model_extra = {"base_url": request.endpoint}
        elif provider.lower() == "azureaifoundry":
            model = "azure_ai/" + request.deployment_name
            model_extra = {"base_url": request.endpoint}
        elif provider.lower() == "custom":
            model = "openai/" + request.deployment_name
            model_extra = {"base_url": request.endpoint}
            if request.auth_token:
                model_extra["extra_headers"] = {"Authorization": request.auth_token}

        model_extra["api_key"] = request.api_key
        try:
            response = litellm.completion(
                messages=messages,
                model=model,
                max_tokens=10,
                temperature=0.0,
                **model_extra,
            )
        except AuthenticationError:
            raise HTTPException(
                status_code=400,
                detail=f"Requested model '{model}' is not available with your api_key and settings.",
            )
        except Exception as e:
            logger.exception(
                f'Failed to check availability of requested model "{model}": {e}\nTraceback:\n{traceback.format_exc()}'
            )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to check availability of requested model '{model}': {e}",
            )

    return Response[ValidateConnectionResponse](
        data=ValidateConnectionResponse(success=True, model=response.model)
    )


@app.post("/models-list", response_model=Response[ModelsListResponse])
async def models_list(request: ModelsListRequest):
    # get_valid_models uses api key set in env, however the list is not dynamically retrieved
    # https://docs.litellm.ai/docs/set_keys#get_valid_models
    # https://github.com/BerriAI/litellm/blob/b9280528d368aced49cb4d287c57cd0b46168cb6/litellm/utils.py#L5705
    # Ultimately just uses litellm.models_by_provider - setting API key is not needed
    lse_provider_to_litellm_provider = {"vertexai": "vertex_ai"}
    provider = request.provider.lower()
    litellm_provider = lse_provider_to_litellm_provider.get(provider, provider)
    valid_models = litellm.models_by_provider[litellm_provider]
    # some providers include the prefix in this list and others don't
    valid_models = [model.replace(f"{litellm_provider}/", "") for model in valid_models]

    return Response[ModelsListResponse](
        data=ModelsListResponse(models_list=valid_models)
    )


@app.post("/estimate-cost", response_model=Response[CostEstimate])
async def estimate_cost(
    request: CostEstimateRequest,
):
    """
    Estimates what it would cost to run inference on the batch of data in
    `request` (using the run params from `request`)

    Args:
        request (CostEstimateRequest): Specification for the inference run to
            make an estimate for, includes:
                agent (adala.agent.Agent): The agent definition, used to get the model
                    and any other params necessary to estimate cost
                prompt (str): The prompt template that will be used for each task
                substitutions (List[Dict]): Mappings to substitute (simply using str.format)

    Returns:
        Response[CostEstimate]: The cost estimate, including the prompt/completion/total costs (in USD)
    """
    prompt = request.prompt
    substitutions = request.substitutions
    agent = request.agent
    provider = request.provider
    runtime = agent.get_runtime()

    try:
        cost_estimates = []
        for skill in agent.skills.skills.values():
            output_fields = (
                list(skill.field_schema.keys()) if skill.field_schema else None
            )
            cost_estimate = await runtime.get_cost_estimate_async(
                prompt=prompt,
                substitutions=substitutions,
                output_fields=output_fields,
                provider=provider,
            )
            cost_estimates.append(cost_estimate)
        total_cost_estimate = sum(
            cost_estimates,
            CostEstimate(
                prompt_cost_usd=None, completion_cost_usd=None, total_cost_usd=None
            ),
        )

    except NotImplementedError as e:
        logger.debug(f"Error estimating cost: {e} {traceback.format_exc()}")
        return Response[CostEstimate](
            data=CostEstimate(
                is_error=True,
                error_type=type(e).__name__,
                error_message=str(e),
            )
        )
    return Response[CostEstimate](data=total_cost_estimate)


class ChatCompletionRequest(BaseModel):
    """
    Request for immediate chat completion.
    """

    messages: List[Dict]
    model: str


class ChatCompletionResponse(BaseModel):
    """
    Response for immediate chat completion following OpenAI chat completion format.
    """

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict
    service_tier: Optional[str] = "default"


def _chat_completion_get_runtime_params(request: Request) -> Dict:
    """
    Get runtime params from the request headers.
    The minimal `runtime_params` must include `api_key` field.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get(
        "authorization"
    )

    if auth_header and auth_header.startswith("Bearer "):
        credentials_payload = auth_header[7:]
        try:
            # Try to decode as base64 and parse as JSON
            decoded_bytes = base64.b64decode(credentials_payload)
            decoded_str = decoded_bytes.decode("utf-8")
            runtime_params = json.loads(decoded_str)

            if "api_key" not in runtime_params:
                raise ValueError("api_key missing from credentials")

            # Remove JWT-related fields as they only needed to check the integrity
            for field in ("exp", "iat", "iss", "sub"):
                runtime_params.pop(field, None)

            return runtime_params

        except Exception:
            # If decoding/parsing fails, treat as plain API key
            return {"api_key": credentials_payload}

    # Fallback to environment variable
    if os.getenv("OPENAI_API_KEY"):
        return {"api_key": os.getenv("OPENAI_API_KEY")}

    raise HTTPException(
        status_code=400,
        detail="No credentials found in the request headers or environment variables",
    )


async def _chat_completion_handle_request(
    chat_request: ChatCompletionRequest, runtime_params: Dict, provider: str
) -> ChatCompletionResponse:
    if isinstance(provider, str) and provider.lower() == "custom":
        # LiteLLM has issues working with Custom OpenAI-compatible API providers
        client = AsyncOpenAI(
            api_key=runtime_params["api_key"], base_url=runtime_params["base_url"]
        )
        response = await client.chat.completions.create(
            messages=chat_request.messages,
            model=chat_request.model,
        )
    else:
        response = await acompletion(
            messages=chat_request.messages,
            model=chat_request.model,
            **runtime_params,
        )
    return ChatCompletionResponse(
        id=response.id,
        object=response.object,
        created=response.created,
        model=response.model,
        choices=[choice.model_dump() for choice in response.choices],
        usage=response.usage.model_dump(),
        service_tier=getattr(response, "service_tier", "default"),
    )


@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: Request, chat_request: ChatCompletionRequest):
    """
    Mimics the OpenAI chat completion API.
    https://platform.openai.com/docs/api-reference/chat/create
    """

    try:
        runtime_params = _chat_completion_get_runtime_params(request)
        if not runtime_params.get("base_url"):
            # Extract base_url from headers if present (due to the OpenAI-compatible API providers)
            runtime_params["base_url"] = request.headers.get("base_url")

        # pop the `model` to ensure it's passed as an input request parameter
        runtime_params.pop("model", None)
        provider = runtime_params.pop("provider", None) or "openai"
        response = await _chat_completion_handle_request(
            chat_request, runtime_params, provider
        )
        return response

    except Exception as e:
        logger.error("Error in chat completion: %s", e)
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


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
    settings = Settings()
    url = settings.redis.to_url()
    kwargs = settings.redis.to_kwargs()
    # set short socket_connect_timeout for ping
    redis_conn = Redis.from_url(url, **kwargs, socket_connect_timeout=1)
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


class ImprovedPromptRequest(BaseModel):
    """
    Request model for improving a prompt.
    """

    agent: Agent
    skill_to_improve: str
    input_variables: Optional[List[str]] = Field(
        default=None,
        description="List of variables available to use in the input template of the skill, in case any exist that are not currently used",
    )
    data: Optional[List[Dict]] = Field(
        default=None,
        description="Batch of data to run the skill on",
    )
    reapply: bool = Field(
        default=False,
        description="Whether to reapply the skill to the data before improving the prompt",
    )
    instructions: Optional[str] = Field(
        default="Improve current prompt",
        description="Instructions for the prompt improvement task",
    )

    @field_validator("agent", mode="after")
    def validate_teacher_runtime(cls, agent: Agent) -> Agent:
        if not isinstance(agent.get_teacher_runtime(), AsyncRuntime):
            raise ValueError("Default teacher runtime must be an AsyncRuntime")
        return agent

    @model_validator(mode="after")
    def set_input_variable_list(self):
        skill = self.agent.skills[self.skill_to_improve]
        if self.input_variables is None:
            self.input_variables = skill.get_input_fields()
        return self


@app.post("/improved-prompt", response_model=Response[ImprovedPromptResponse])
async def improved_prompt(request: ImprovedPromptRequest):
    """
    Improve a given prompt using the specified model and variables.

    Args:
        request (ImprovedPromptRequest): The request model for improving a prompt.

    Returns:
        Response: Response model for prompt improvement skill
    """
    improved_prompt_response = await request.agent.arefine_skill(
        skill_name=request.skill_to_improve,
        input_variables=request.input_variables,
        data=request.data,
        reapply=request.reapply,
        instructions=request.instructions,
    )

    return Response[ImprovedPromptResponse](
        success=not isinstance(improved_prompt_response.output, ErrorResponseModel),
        data=improved_prompt_response,
    )


class ModelMetadataRequestItem(BaseModel):
    provider: str
    model_name: str
    auth_info: Optional[Dict[str, str]] = None


class ModelMetadataRequest(BaseModel):
    models: List[ModelMetadataRequestItem]


class ModelMetadataResponse(BaseModel):
    model_metadata: Dict[str, Dict]


@app.post("/model-metadata", response_model=Response[ModelMetadataResponse])
async def model_metadata(request: ModelMetadataRequest):
    from adala.utils.model_info_utils import get_model_info

    resp = {
        "model_metadata": {
            item.model_name: get_model_info(**item.model_dump())
            for item in request.models
        }
    }
    return Response[ModelMetadataResponse](success=True, data=resp)


# Simple cleanup handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the server shuts down"""
    logger.info("Shutting down server...")

    # Clean up worker pool API producer
    try:
        from server.worker_pool.api import cleanup_kafka_producer

        await cleanup_kafka_producer()
        logger.info("Worker pool API producer cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up worker pool API producer: {e}")

    logger.info("Server shutdown complete")


if __name__ == "__main__":
    # for debugging
    uvicorn.run("app:app", host="0.0.0.0", port=30001)
