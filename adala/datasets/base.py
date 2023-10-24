from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator
from typing import List, Optional, Any, Dict, Union

from adala.utils.internal_data import InternalDataFrame


class Dataset(BaseModel, ABC):
    """
    Abstract base class representing a dataset.

    Provides methods to interact with and obtain information about datasets. 
    Concrete implementations should provide functionality for batch iteration, 
    getting dataset size, and displaying dataset information.
    """

    @abstractmethod
    def batch_iterator(self, batch_size: int = 100) -> InternalDataFrame:
        """
        Yields batches of data records from the dataset.
        
        Args:
            batch_size (int, optional): Size of each batch to be yielded. Defaults to 100.
            
        Returns:
            InternalDataFrame: A data frame containing a batch of records.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Provides the number of records in the dataset.
        
        Returns:
            int: Total number of records in the dataset.
        """

    @abstractmethod
    def info(self) -> None:
        """
        Displays information about the dataset.
        """


class BlankDataset(Dataset):
    """
    Represents an empty dataset with no records.

    This class can be used in situations where a dataset is required, 
    but no actual data is available or needed. 
    All methods return defaults representing an empty state.
    """

    def batch_iterator(self, batch_size: int = 100) -> InternalDataFrame:
        """
        Yields an empty data frame as there are no records in a blank dataset.
        
        Args:
            batch_size (int, optional): This argument is ignored for BlankDataset. Defaults to 100.
            
        Returns:
            InternalDataFrame: An empty data frame.
        """
        
        return InternalDataFrame()

    def __len__(self) -> int:
        """
        Provides the number of records in the blank dataset (which is always 0).
        
        Returns:
            int: Total number of records in the dataset (0 for BlankDataset).
        """
        
        return 0

    def info(self) -> None:
        """
        Displays information about the blank dataset.
        """
        
        print('Blank dataset')
