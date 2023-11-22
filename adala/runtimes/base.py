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

    @model_validator(mode='after')
    def init_runtime(self) -> 'Runtime':
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

        Returns:
            InternalDataFrame: The processed batch.
        """
        output = batch.progress_apply(
            self.record_to_record,
            axis=1,
            result_type='expand',
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
