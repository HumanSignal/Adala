from tqdm import tqdm
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import List, Dict, Optional, Tuple, Any, Callable
from adala.utils.internal_data import InternalDataFrame, InternalSeries

tqdm.pandas()


class Runtime(BaseModel, ABC):
    """
    Base class representing a generic runtime environment.

    Attributes:
        verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
    """

    verbose: bool = False
    batch_size: Optional[int] = None

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
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
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

        Returns:
            Dict[str, str]: The processed record.
        """

    def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        """
        Processes a record.

        Args:
            batch (InternalDataFrame): The batch to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            output_template (str): The output template.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.

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
        )
        return output

    def record_to_batch(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        output_batch_size: int = 1,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        """
        Processes a record and return a batch.

        Args:
            record (Dict[str, str]): The record to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            output_template (str): The output template.
            output_batch_size (int): The batch size for the output. Defaults to 1.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.

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
        )


class AsyncRuntime(BaseModel, ABC):
    """Async version of runtime that uses asyncio to process batch of records."""

    verbose: bool = False
    batch_size: int = 100

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

        Returns:
            Dict[str, str]: The processed record.
        """

    @abstractmethod
    async def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        """
        Processes a record.

        Args:
            batch (InternalDataFrame): The batch to process.
            input_template (str): The input template.
            instructions_template (str): The instructions template.
            output_template (str): The output template.
            extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
            field_schema (Optional[Dict]): Field JSON schema to use in the templates. Defaults to all fields are strings,
                i.e. analogous to {"field_n": {"type": "string"}}.
            instructions_first (bool): Whether to put instructions first. Defaults to True.

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
        )
        return output

    async def get_next_batch(self, data_iterator, batch_size: Optional[int]) -> InternalDataFrame:
        if batch_size is None:
            batch_size = self.optimal_batch_size
        batch = []
        try:
            for _ in range(batch_size):
                data = await anext(data_iterator, None)
                if data is None:  # This checks if the iterator is exhausted
                    break
                batch.append(data)
        except StopAsyncIteration:
            pass
        return InternalDataFrame(batch)


_runtimes_register = {}


def register_runtime(type_name, type_class):
    global _runtimes_register

    if type_name in _runtimes_register:
        raise ValueError(f"Runtime {type_name} already registered. Available runtimes: {list(_runtimes_register.keys())}")

    _runtimes_register[type_name] = type_class


def create_runtime(type_name, **kwargs):
    if type_name not in _runtimes_register:
        raise ValueError(f"Runtime {type_name} not found. Available runtimes: {list(_runtimes_register.keys())}")
    return _runtimes_register[type_name](**kwargs)
