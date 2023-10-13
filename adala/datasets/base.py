from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Union

RawRecord = Dict[str, Any]
RawRecords = List[RawRecord]

# Internal data tables representation. Replace this with Dask or Polars in the future.
InternalDataFrame = pd.DataFrame


class Dataset(BaseModel, ABC):
    """
    Base class for original datasets.
    """
    ground_truth_column: str = 'ground_truth'

    @abstractmethod
    def template_string_batches(
        self,
        template: str,
        instructions: str,
        batch_size: int = 100
    ):
        """
        Yield batches of template strings for given template and instruction.
        """

    @abstractmethod
    def make_new_with_index(self, records: RawRecords) -> InternalDataFrame:
        """
        Return new dataset with the same index as original dataset, but with new records.
        """

    @abstractmethod
    def assign(self, records: Union[RawRecords, InternalDataFrame], inplace=False) -> Optional[InternalDataFrame]:
        """
        Assign new records to the original dataset.
        If inplace is True, modify the original dataset.
        """

    @abstractmethod
    def assign_columns_match(
        self, column_a: str, column_b: str, inplace: bool = False, output_column_name: str = 'match'
    ) -> Optional["Dataset"]:
        """
        Assign new column to the original dataset with True/False values
        indicating whether the values in column_a and column_b match.
        """

    @abstractmethod
    def simple_select(self, column_name: str, value: Any) -> "Dataset":
        """
        Return subset of the original dataset where column_name == value.
        """

    @abstractmethod
    def get_ground_truth(self) -> InternalDataFrame:
        """
        Return ground truth subset if available.
        """

    @abstractmethod
    def get_column_stats(self, column_name: str) -> Dict:
        """
        Return basic statistics for a given column.
        """