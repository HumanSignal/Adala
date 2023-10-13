import pandas as pd
from typing import List, Dict, Any, Union, Optional, Callable
from .base import Dataset, RawRecords
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
        instruction: str,
        batch_size: int = 100
    ):
        for i in range(0, len(self.df), batch_size):
            transformed_batch = self.df.iloc[i:i+batch_size].apply(
                lambda row: template.format(instruction=instruction, input=row.to_json()),
                axis=1)
            yield transformed_batch.tolist()

    def make_new(self, records: RawRecords) -> "PandasDataFrame":
        index = self.df.index
        if len(records) == 0:
            return PandasDataFrame(df=pd.DataFrame(index=index))

        return PandasDataFrame(df=pd.DataFrame(records, index=index))

    def assign(self, records: Union[RawRecords, "PandasDataFrame"], inplace=False) -> Optional["PandasDataFrame"]:
        if isinstance(records, PandasDataFrame):
            new_df = records.df.loc(self.df.index)
        else:
            new_df = pd.DataFrame(records, index=self.df.index)
        if inplace:
            self.df = pd.concat([self.df, new_df], axis=1)
        else:
            return PandasDataFrame(df=new_df)

    def assign_columns_match(
        self, column_a: str, column_b: str, inplace: str = False, output_column_name: str = 'match'
    ) -> Optional["Dataset"]:
        dfa = self.df[self.df[column_a].notna()]
        dfb = self.df[self.df[column_b].notna()]
        common_indices = dfa.index.isin(dfb.index)

        # TODO: implement more sophisticated evaluation beyond simple equality
        match = dfa.loc[common_indices, column_a] == dfb.loc[common_indices, column_b]

        if inplace:
            self.df[output_column_name] = float('nan')
            self.df.loc[common_indices, output_column_name] = match
        else:
            return PandasDataFrame(df=pd.DataFrame({output_column_name: match}))

    def simple_select(self, column_name: str, value: Any) -> "PandasDataFrame":
        return PandasDataFrame(df=self.df[self.df[column_name] == value])

    def get_ground_truth(self) -> "PandasDataFrame":
        return PandasDataFrame(df=self.df[self.df[self.ground_truth_column].notna()])

    def get_column_stats(self, column_name):
        basic_distribution = self.df[column_name].value_counts().to_dict()
        basic_distribution['total'] = len(self.df)
        return basic_distribution
