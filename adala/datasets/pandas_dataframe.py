import pandas as pd
from typing import List, Dict, Any, Union, Optional, Callable
from .base import Dataset, RawRecords, InternalDataFrame
from pydantic import validator


class PandasDataFrame(Dataset):
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @validator('df', pre=True)
    def validate_df(cls, df: pd.DataFrame, values) -> pd.DataFrame:
        ground_truth_column = values.get('ground_truth_column')
        if ground_truth_column not in df.columns:
            df[ground_truth_column] = float('nan')

        return df

    def __len__(self):
        return len(self.df)

    def apply_template(self, template: str) -> "PandasDataFrame":
        return self.df.apply(
            func=lambda row: template.format(**row.to_dict()),
            axis=1
        )

    def sample(self, n: int) -> "PandasDataFrame":
        return PandasDataFrame(df=self.df.sample(n))

    def template_string_batches(
        self,
        template: str,
        instructions: str,
        batch_size: int = 100
    ):
        for i in range(0, len(self.df), batch_size):
            transformed_batch = self.df.iloc[i:i+batch_size].apply(
                lambda row: template.format(instructions=instructions, input=row.to_json()), axis=1)
            yield transformed_batch.tolist()

    def make_new_with_index(self, records: RawRecords) -> InternalDataFrame:
        index = self.df.index
        if len(records) == 0:
            return InternalDataFrame(index=index)

        return InternalDataFrame(records, index=index)

    def assign(self, records: Union[RawRecords, "PandasDataFrame"], inplace=False) -> Optional["PandasDataFrame"]:
        if isinstance(records, PandasDataFrame):
            new_df = records.df.loc(self.df.index)
        else:
            new_df = pd.DataFrame(records, index=self.df.index)
        if inplace:
            self.df = pd.concat([self.df, new_df], axis=1)
        else:
            return PandasDataFrame(df=new_df)

    def simple_select(self, column_name: str, value: Any) -> "PandasDataFrame":
        return PandasDataFrame(df=self.df[self.df[column_name] == value])

    def get_ground_truth(self) -> InternalDataFrame:
        return self.df[self.df[self.ground_truth_column].notna()]

    def get_column_stats(self, column_name):
        basic_distribution = self.df[column_name].value_counts().to_dict()
        basic_distribution['total'] = len(self.df)
        return basic_distribution
