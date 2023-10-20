from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Union, Iterable

RawRecord = Dict[str, Any]
RawRecords = List[RawRecord]

# Internal data tables representation. Replace this with Dask or Polars in the future.
InternalDataFrame = pd.DataFrame


def InternalDataFrame_encoder(df: InternalDataFrame) -> List:
    return df.to_dict(orient='records')


class Dataset(BaseModel, ABC):
    """
    Base class for original datasets.
    """

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
