import pandas as pd
from typing import List, Dict, Any, Union, Iterable, Optional, Type

Record = Dict[str, Any]

# Use pandas DataFrame for internal data representation
class DataTable(pd.DataFrame):
    """
    A simple wrapper around pandas DataFrame to provide common batch processing methods.
    This provides a direct interface for tabular data handling with LLMs.
    """
    
    @classmethod
    def from_records(cls, data: List[Dict]) -> 'DataTable':
        """Create a DataTable from a list of dictionaries."""
        return cls(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'DataTable':
        """Create a DataTable from a pandas DataFrame."""
        return cls(df)
    
    def to_records(self) -> List[Dict]:
        """Convert to a list of dictionaries."""
        return self.to_dict(orient="records")
    
    @classmethod
    def concat(cls, dfs: Iterable['DataTable'], **kwargs) -> 'DataTable':
        """Concatenate multiple DataTables."""
        return cls(pd.concat(dfs, **kwargs))

# For backward compatibility
InternalDataFrame = DataTable
InternalSeries = pd.Series

def InternalDataFrame_encoder(df: InternalDataFrame) -> List:
    return df.to_records()

def InternalDataFrameConcat(
    dfs: Iterable[InternalDataFrame], **kwargs
) -> InternalDataFrame:
    """
    Concatenate dataframes.

    Args:
        dfs (Iterable[InternalDataFrame]): The dataframes to concatenate.

    Returns:
        InternalDataFrame: The concatenated dataframe.
    """
    return DataTable.concat(dfs, **kwargs)
