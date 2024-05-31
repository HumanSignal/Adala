from typing import Optional
import logging
import json
from abc import abstractmethod
from pydantic import BaseModel, Field, computed_field, ConfigDict, model_validator

from adala.utils.registry import BaseModelInRegistry


logger = logging.getLogger(__name__)

try:
    from label_studio_sdk import Client as LSEClient
except ImportError:
    logger.warning(
        "Label Studio SDK not found. LSEHandler will not be available. Run `poetry install --with label-studio` to fix"
    )

    class LSEClient:
        def __init__(self, *args, **kwargs):
            logger.error(
                "Label Studio SDK not found. LSEHandler is not available. Run `poetry install --with label-studio` to fix"
            )


class ResultHandler(BaseModelInRegistry):
    """
    Abstract base class for a result handler.
    This is a callable that is instantiated in `/submit-streaming` with any arguments that are needed, and then is called on each batch of results when it is finished being processed by the Agent (it consumes from the Kafka topic that the Agent produces to).

    It can be used as a connector to load results into a file or external service. If a ResultHandler is not used, the results will be discarded.

    Subclasses must implement the `__call__` method.

    The BaseModelInRegistry base class implements a factory pattern, allowing the "type" parameter to specify which subclass of ResultHandler to instantiate. For example:
    ```json
    result_handler: {
        "type": "DummyHandler",
        "other_model_field": "other_model_value",
        ...
    }
    ```
    """

    @abstractmethod
    def __call__(self, result_batch: list[dict]) -> None:
        """
        Callable to do something with a batch of results.
        """
        pass


class DummyHandler(ResultHandler):
    """
    Dummy handler to test streaming output flow
    """

    def __call__(self, batch):
        logger.info(f"\n\nHandler received batch: {batch}\n\n")


class LSEBatchItem(BaseModel):
    """
    The output for a single Task in an LSE Project.
    A batch of these consumed from the kafka output queue is the expected input for LSEHandler.__call__
    """

    model_config = ConfigDict(
        # omit fields from the input task besides task_id, the LSE /submit-batch endpoint doesn't use these and it'd be a waste of network bandwidth since they can be large
        extra="ignore",
        # guard against name collisions with other input fields
        allow_population_by_field_name=False,
    )

    task_id: int
    output: str
    # TODO handle in DIA-1122
    # we don't need to use reserved names anymore here because they're not in a DataFrame, but a structure with proper typing available
    error: bool = Field(False, alias="_adala_error")
    message: Optional[str] = Field(None, alias="_adala_message")
    details: Optional[str] = Field(None, alias="_adala_details")

    @model_validator(mode="after")
    def check_error_consistency(self):
        has_error = self.error
        message = self.message
        details = self.details

        if has_error and (message is None or details is None):
            raise ValueError("_adala_error is set, but has no error message or details")
        elif not has_error and (message is not None or details is not None):
            raise ValueError(
                "_adala_error is unset, but has an error message or details"
            )

        return self


class LSEHandler(ResultHandler):
    """
    Handler to use the Label Studio SDK to load a batch of results back into a Label Studio project
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    api_key: str
    url: str
    modelrun_id: int

    @computed_field
    def client(self) -> LSEClient:
        _client = LSEClient(
            api_key=self.api_key,
            url=self.url,
        )
        # Need this to make POST requests using the SDK client
        # TODO headers can only be set in this function, since client is a computed field. Need to rethink approach if we make non-POST requests, should probably just make a PR in label_studio_sdk to allow setting this in make_request()
        _client.headers.update(
            {
                "accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        return _client

    @model_validator(mode="after")
    def ready(self):
        conn = self.client.check_connection()
        assert conn["status"] == "UP", "Label Studio is not available"

        return self

    def __call__(self, result_batch: list[LSEBatchItem]):
        logger.info(f"\n\nHandler received batch: {result_batch}\n\n")

        # coerce dicts to LSEBatchItems for validation
        result_batch = [LSEBatchItem(**record) for record in result_batch]

        # omit failed tasks for now
        # TODO handle in DIA-1122
        result_batch = [record for record in result_batch if not record.error]

        # coerce back to dicts for sending
        result_batch = [record.dict() for record in result_batch]

        self.client.make_request(
            "POST",
            "/api/model-run/batch-predictions",
            data=json.dumps(
                {
                    "modelrun_id": self.modelrun_id,
                    "results": result_batch,
                }
            ),
        )
