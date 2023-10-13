from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Union

RawRecord = Dict[str, Any]
RawRecords = List[RawRecord]


class Dataset(BaseModel, ABC):
    """
    Base class for original datasets.
    """
    ground_truth_column: str = 'ground_truth'

    @abstractmethod
    def template_string_batches(
        self,
        template: str,
        instruction: str,
        batch_size: int = 100
    ):
        """
        Yield batches of template strings for given template and instruction.
        """

    @abstractmethod
    def make_new_with_index(self, records: RawRecords) -> "Dataset":
        """
        Return new dataset with the same index as original dataset, but with new records.
        """

    @abstractmethod
    def assign(self, records: Union[RawRecords, "Dataset"], inplace=False) -> Optional["Dataset"]:
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
    def get_ground_truth(self) -> "Dataset":
        """
        Return ground truth subset if available.
        """

    @abstractmethod
    def get_column_stats(self, column_name: str) -> Dict:
        """
        Return basic statistics for a given column.
        """