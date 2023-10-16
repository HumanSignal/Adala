from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Union

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
    ground_truth_column: str = 'ground_truth'

    @abstractmethod
    def batch_iterator(self, batch_size: int = 100) -> List[RawRecords]:
        """
        Yield batches of data.
        """

    @abstractmethod
    def make_new_with_index(self, records: RawRecords) -> InternalDataFrame:
        """
        Return new dataset with the same index as original dataset, but with new records.
        """

    @abstractmethod
    def get_ground_truth(self) -> InternalDataFrame:
        """
        Return ground truth subset if available.
        """
