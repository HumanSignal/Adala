import pandas as pd
from typing import List, Dict, Any, Union, Optional, Callable, Iterable
from .base import Dataset, RawRecords, InternalDataFrame
from pydantic import model_validator


class DataFrameDataset(Dataset):
    df: InternalDataFrame
    ground_truth_column: str = 'ground_truth'

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def validate_ground_truth_column_existence(self):
        if self.ground_truth_column not in self.df.columns:
            self.df[self.ground_truth_column] = float('nan')
        return self

    def __len__(self):
        return len(self.df)

    def batch_iterator(self, batch_size: int = 100) -> Iterable[InternalDataFrame]:
        for i in range(0, len(self.df), batch_size):
            batch = self.df.iloc[i:i+batch_size]
            # if ground truth column is not present, we have to drop it to avoid aka "leakage"
            # use `get_ground_truth(batch)` to get ground truth subset
            batch = batch.drop(columns=[self.ground_truth_column])
            yield batch

    def get_ground_truth(self, batch: Optional[InternalDataFrame] = None) -> InternalDataFrame:
        if batch is None:
            return self.df[self.df[self.ground_truth_column].notna()]
        else:
            return self.df[self.df[self.ground_truth_column].notna() & self.df.index.isin(batch.index)]
