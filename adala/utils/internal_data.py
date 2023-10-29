import pandas as pd
from typing import List, Dict, Any, Union, Iterable

RawRecord = Dict[str, Any]
RawRecords = List[RawRecord]

# Internal data tables representation. Replace this with Dask or Polars in the future.
InternalDataFrame = pd.DataFrame
InternalSeries = pd.Series


def InternalDataFrame_encoder(df: InternalDataFrame) -> List:
    return df.to_dict(orient='records')


def InternalDataFrameConcat(dfs: Iterable[InternalDataFrame], **kwargs) -> InternalDataFrame:
    return pd.concat(dfs, **kwargs)
