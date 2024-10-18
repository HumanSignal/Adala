import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from adala.utils.internal_data import InternalDataFrame
from adala.utils.registry import BaseModelInRegistry
from pandarallel import pandarallel
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)
tqdm.pandas()


class CostEstimate(BaseModel):
    prompt_cost_usd: Optional[float] = None
    completion_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    is_error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        # if either has an error, it takes precedence
        if self.is_error:
            return self
        if other.is_error:
            return other

        def _safe_add(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
            if lhs is None and rhs is None:
                return None
            _lhs = lhs or 0.0
            _rhs = rhs or 0.0
            return _lhs + _rhs

        prompt_cost_usd = _safe_add(self.prompt_cost_usd, other.prompt_cost_usd)
        completion_cost_usd = _safe_add(
            self.completion_cost_usd, other.completion_cost_usd
        )
        total_cost_usd = _safe_add(self.total_cost_usd, other.total_cost_usd)
        return CostEstimate(
            prompt_cost_usd=prompt_cost_usd,
            completion_cost_usd=completion_cost_usd,
            total_cost_usd=total_cost_usd,
        )


class Runtime(BaseModelInRegistry):
    """
    Base class representing a generic runtime environment.

    Attributes:
        verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
        batch_size (Optional[int]): The batch size to use for processing records. Defaults to None.
        concurrency (Optional[int]): The number of parallel processes to use for processing records. Defaults to 1.
                                    Note that when parallel processing is used, the memory footprint will be doubled compared to sequential processing.
        response_model (Optional[Type[BaseModel]]): The response model to use for processing records. Defaults to None.
                                                    If set, the response will be generated according to this model and `output_template` and `field_schema` fields will be ignored.
                                                    Note, explicitly providing ResponseModel will be the default behavior for all runtimes in the future.
    """

    verbose: bool = False
    batch_size: Optional[int] = None
    concurrency: Optional[int] = Field(default=1, alias="concurrent_clients")

    @model_validator(mode="after")
    def init_runtime(self) -> "Runtime":
        """Initializes the runtime.

        This method should be used to validate and potentially initialize the runtime instance.

        Returns:
            Runtime: The initialized runtime instance.
        """
        return self

    @abstractmethod
    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        output_template: Optional[
            str
        ] = None,  # TODO: deprecated in favor of response_model, can be removed
    ) -> Dict[str, str]:
        """
        Processes a record.

        Args:
            record (Dict[str, str]): The record to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            response_model (Type[BaseModel]): The response model to use for processing records.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.
            output_template (str): The output template. Deprecated.

        Returns:
            Dict[str, str]: The processed record.
        """

    def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        output_template: Optional[
            str
        ] = None,  # TODO: deprecated in favor of response_model, can be removed
    ) -> InternalDataFrame:
        """
        Processes a record.
        It supports parallel processing of the batch:
         - when the `concurrency` is set to -1 (using all available CPUs),
         - when the `concurrency` is set to 1 (sequential processing),
         - when the `concurrency` is set to a fixed number of CPUs.
        Please note that parallel processing doubles the memory footprint compared to sequential processing.

        Args:
            batch (InternalDataFrame): The batch to process.
            input_template (str): The input template.
            instructions_template (str): The instructions' template.
            response_model (Type[BaseModel]): The response model to use for processing records.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.
            output_template (str): The output template. Deprecated.
        Returns:
            InternalDataFrame: The processed batch.
        """
        if self.concurrency == -1:
            # run batch processing each row in a parallel way, using all available CPUs
            logger.info("Running batch processing in parallel using all available CPUs")
            pandarallel.initialize(progress_bar=self.verbose)
            apply_func = batch.parallel_apply
        elif self.concurrency == 1:
            # run batch processing each row in a sequential way
            logger.info("Running batch processing sequentially")
            if self.verbose:
                apply_func = batch.progress_apply
            else:
                apply_func = batch.apply
        elif self.concurrency > 1:
            # run batch processing each row in a parallel way, using a fixed number of CPUs
            logger.info(
                f"Running batch processing in parallel using {self.concurrency} CPUs"
            )
            # Warning: parallel processing doubles the memory footprint compared to sequential processing
            # read more about https://nalepae.github.io/pandarallel/
            pandarallel.initialize(
                nb_workers=self.concurrency, progress_bar=self.verbose
            )
            apply_func = batch.parallel_apply
        else:
            raise ValueError(f"Invalid concurrency value: {self.concurrency}")

        output = apply_func(
            self.record_to_record,
            axis=1,
            result_type="expand",
            input_template=input_template,
            instructions_template=instructions_template,
            output_template=output_template,
            extra_fields=extra_fields,
            field_schema=field_schema,
            instructions_first=instructions_first,
            response_model=response_model,
        )
        return output

    def record_to_batch(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        output_batch_size: int = 1,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        output_template: Optional[
            str
        ] = None,  # TODO: deprecated in favor of response_model, can be removed
    ) -> InternalDataFrame:
        """
        Processes a record and return a batch.

        Args:
            record (Dict[str, str]): The record to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            response_model (Optional[Type[BaseModel]]): The response model to use for processing records. Defaults to None.
            output_batch_size (int): The batch size for the output. Defaults to 1.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.
            output_template (str): The output template. Deprecated.

        Returns:
            InternalDataFrame: The processed batch.
        """
        batch = InternalDataFrame([record] * output_batch_size)
        return self.batch_to_batch(
            batch=batch,
            input_template=input_template,
            instructions_template=instructions_template,
            output_template=output_template,
            extra_fields=extra_fields,
            field_schema=field_schema,
            instructions_first=instructions_first,
            response_model=response_model,
        )

    def get_cost_estimate(
        self, prompt: str, substitutions: List[Dict], output_fields: Optional[List[str]]
    ) -> CostEstimate:
        raise NotImplementedError("This runtime does not support cost estimates")


class AsyncRuntime(Runtime):
    """Async version of runtime that uses asyncio to process batch of records."""

    @abstractmethod
    async def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, str]:
        """
        Processes a record.

        Args:
            record (Dict[str, str]): The record to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            output_template (str): The output template.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.
            response_model (Optional[Type[BaseModel]]): The response model to use for processing records. Defaults to None.
                                                        If set, the response will be generated according to this model and `output_template` and `field_schema` fields will be ignored.
                                                        Note, explicitly providing ResponseModel will be the default behavior for all runtimes in the future.

        Returns:
            Dict[str, str]: The processed record.
        """

    @abstractmethod
    async def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        response_model: Type[BaseModel],
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
        output_template: Optional[
            str
        ] = None,  # TODO: deprecated in favor of response_model, can be removed
    ) -> InternalDataFrame:
        """
        Processes a record.

        Args:
            batch (InternalDataFrame): The batch to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            response_model (Optional[Type[BaseModel]]): The response model to use for processing records.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.
            output_template (str): The output template. Deprecated.

        Returns:
            InternalDataFrame: The processed batch.
        """
        output = batch.progress_apply(
            self.record_to_record,
            axis=1,
            result_type="expand",
            input_template=input_template,
            instructions_template=instructions_template,
            output_template=output_template,
            extra_fields=extra_fields,
            field_schema=field_schema,
            instructions_first=instructions_first,
            response_model=response_model,
        )
        return output
