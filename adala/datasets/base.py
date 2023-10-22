from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator
from typing import List, Optional, Any, Dict, Union

from adala.utils.internal_data import InternalDataFrame


class Dataset(BaseModel, ABC):
    """
    Base class for original datasets.
    """
    input_data_field: Optional[str] = None

    @abstractmethod
    def batch_iterator(self, batch_size: int = 100) -> InternalDataFrame:
        """
        Yield batches of data records
        """

    @abstractmethod
    def get_ground_truth(self, batch: Optional[InternalDataFrame]) -> InternalDataFrame:
        """
        Return ground truth subset if available.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return number of records in dataset.
        """

    @abstractmethod
    def info(self) -> None:
        """
        Print dataset information.
        """


class BlankDataset(Dataset):

    def batch_iterator(self, batch_size: int = 100) -> InternalDataFrame:
        return InternalDataFrame()

    def get_ground_truth(self, batch: Optional[InternalDataFrame]) -> InternalDataFrame:
        return InternalDataFrame()

    def __len__(self) -> int:
        return 0

    def info(self) -> None:
        print('Blank dataset')
