from typing import Optional, List, Dict
import json
from abc import abstractmethod
from pydantic import BaseModel, Field, computed_field, ConfigDict, model_validator
import csv

from adala.utils.registry import BaseModelInRegistry
from server.utils import init_logger

logger = init_logger(__name__)

try:
    from label_studio_sdk import LabelStudio as LSEClient
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
        logger.debug(f"\n\nHandler received batch: {batch}\n\n")


class LSEBatchItem(BaseModel):
    """
    The output for a single Task in an LSE Project.
    A batch of these consumed from the kafka output topic is the expected input for LSEHandler.__call__
    """

    model_config = ConfigDict(
        # omit fields from the input task besides task_id, the LSE /batch-predictions endpoint doesn't use these and it'd be a waste of network bandwidth since they can be large
        extra="ignore",
        # guard against name collisions with other input fields
        populate_by_name=False,
    )

    task_id: int
    # TODO this field no longer populates if there was an error, so validation fails without a default - should probably split this item into 3 different constructors corresponding to new internal adala objects (or just reuse those objects)
    output: Optional[Dict] = None
    # we don't need to use reserved names anymore here because they're not in a DataFrame, but a structure with proper typing available
    error: bool = Field(False, alias="_adala_error")
    message: Optional[str] = Field(None, alias="_adala_message")
    details: Optional[str] = Field(None, alias="_adala_details")

    prompt_tokens: int = Field(alias="_prompt_tokens")
    completion_tokens: int = Field(alias="_completion_tokens")

    # these can fail to calculate
    prompt_cost_usd: Optional[float] = Field(alias="_prompt_cost_usd")
    completion_cost_usd: Optional[float] = Field(alias="_completion_cost_usd")
    total_cost_usd: Optional[float] = Field(alias="_total_cost_usd")
    message_counts: Optional[Dict[str, int]] = Field(
        alias="_message_counts", default_factory=dict
    )
    inference_time: Optional[float] = Field(alias="_inference_time")

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

    @classmethod
    def from_result(cls, result: Dict) -> "LSEBatchItem":
        """
        Prepare a result for processing by the handler:
        - extract error, message and detail if result is a failed prediction
        - otherwise, put the result payload to the output field
        """
        # Copy system fields
        prepared_result = {
            k: v
            for k, v in result.items()
            if k
            in (
                "task_id",
                "_adala_error",
                "_adala_message",
                "_adala_details",
                "_prompt_tokens",
                "_completion_tokens",
                "_prompt_cost_usd",
                "_completion_cost_usd",
                "_total_cost_usd",
                "_message_counts",
                "_inference_time",
            )
        }

        # Normalize results if they contain NaN
        if result.get("_adala_error") != result.get("_adala_error"):
            prepared_result["_adala_error"] = False
        if result.get("_adala_message") != result.get("_adala_message"):
            prepared_result["_adala_message"] = None
        if result.get("_adala_details") != result.get("_adala_details"):
            prepared_result["_adala_details"] = None

        # filter out the rest of custom fields
        prepared_result["output"] = {
            k: v for k, v in result.items() if k not in prepared_result
        }

        logger.debug("Prepared result: %s", prepared_result)

        return cls(**prepared_result)


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
            base_url=self.url,
        )

        return _client

    @model_validator(mode="after")
    def ready(self):
        # Use versions endpoint to verify connection to LS instance
        assert self.client.versions.get()

        return self

    def prepare_errors_payload(self, error_batch):
        transformed_errors = []
        for error in error_batch:
            error = error.dict()
            transformed_error = {
                "task_id": error["task_id"],
                "message": error["details"] if "details" in error else "",
                "error_type": error["message"] if "message" in error else "",
            }
            transformed_errors.append(transformed_error)

        return transformed_errors

    def __call__(self, result_batch: list[Dict]):
        logger.debug(f"\n\nHandler received batch: {result_batch}\n\n")
        logger.info("LSEHandler received batch")

        # coerce dicts to LSEBatchItems for validation
        norm_result_batch = [
            LSEBatchItem.from_result(result) for result in result_batch
        ]

        result_batch = [record for record in norm_result_batch if not record.error]
        error_batch = [record for record in norm_result_batch if record.error]

        # coerce back to dicts for sending
        result_batch = [record.dict() for record in result_batch]
        if result_batch:
            num_predictions = len(result_batch)
            logger.info(f"LSEHandler sending {num_predictions} predictions to LSE")
            self.client.prompts.batch_predictions(
                modelrun_id=self.modelrun_id,
                results=result_batch,
                num_predictions=num_predictions
            )
            logger.info(f"LSEHandler sent {num_predictions} predictions to LSE")
        else:
            logger.error(
                f"No valid results to send to LSE for modelrun_id {self.modelrun_id}"
            )

        # Send failed predictions back to LSE
        if error_batch:
            error_batch = self.prepare_errors_payload(error_batch)
            num_failed_predictions = len(error_batch)
            logger.info(
                f"LSEHandler sending {num_failed_predictions} failed predictions to LSE"
            )
            self.client.prompts.batch_failed_predictions(
                modelrun_id=self.modelrun_id,
                failed_predictions=error_batch,
                num_failed_predictions=num_failed_predictions
            )
            logger.info(
                f"LSEHandler sent {num_failed_predictions} failed predictions to LSE"
            )
        else:
            logger.debug(f"No errors to send to LSE for modelrun_id {self.modelrun_id}")


class CSVHandler(ResultHandler):
    """
    Handler to write a batch of results to a CSV file
    """

    output_path: str
    columns: Optional[list[str]] = None

    @model_validator(mode="after")
    def write_header(self):
        if self.columns is None:
            self.columns = list(LSEBatchItem.model_fields.keys())

        with open(self.output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

        return self

    def __call__(self, result_batch: List[Dict]):
        logger.debug(f"\n\nHandler received batch: {result_batch}\n\n")

        # coerce dicts to LSEBatchItems for validation
        norm_result_batch = [
            LSEBatchItem.from_result(result) for result in result_batch
        ]

        # open and write to file
        with open(self.output_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerows([record.dict() for record in norm_result_batch])
