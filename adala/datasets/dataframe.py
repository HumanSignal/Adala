from typing import Iterable
from .base import Dataset
from adala.utils.internal_data import InternalDataFrame
from pydantic import Field


class DataFrameDataset(Dataset):
    """
    Represents a dataset backed by an internal data frame.

    Provides methods to interact with and obtain information about the dataset stored
    as an internal data frame. This class wraps around `InternalDataFrame` to make it 
    compatible with the dataset abstraction.

    Attributes:
        df (InternalDataFrame): The internal data frame storing the dataset.
    """
    
    df: InternalDataFrame = Field(default_factory=InternalDataFrame)

    class Config:
        arbitrary_types_allowed = True

    def __len__(self):
        """
        Provides the number of records in the dataset.
        
        Returns:
            int: Total number of records in the dataset.
        """
        
        return len(self.df)

    def batch_iterator(self, batch_size: int = 100) -> Iterable[InternalDataFrame]:
        """
        Yields batches of data records from the dataset.
        
        Args:
            batch_size (int, optional): Size of each batch to be yielded. Defaults to 100.
            
        Yields:
            Iterable[InternalDataFrame]: An iterator that yields data frames containing batches of records.
        """
        
        for i in range(0, len(self.df), batch_size):
            yield self.df.iloc[i:i+batch_size]

    def info(self) -> None:
        """
        Displays information (statistical description) about the dataset.
        """
        
        print(self.df.describe())
