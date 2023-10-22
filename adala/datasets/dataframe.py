from typing import Iterable
from .base import Dataset
from adala.utils.internal_data import InternalDataFrame


class DataFrameDataset(Dataset):
    df: InternalDataFrame

    class Config:
        arbitrary_types_allowed = True

    def __len__(self):
        return len(self.df)

    def batch_iterator(self, batch_size: int = 100) -> Iterable[InternalDataFrame]:
        for i in range(0, len(self.df), batch_size):
            yield self.df.iloc[i:i+batch_size]

    def info(self) -> None:
        print(self.df.describe())