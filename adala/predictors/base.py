import pandas as pd
import difflib
import json

from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel, validator, root_validator
from typing import Optional, List
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.llms import BaseLLM


class Predictor(BaseModel, ABC):
    """
    Base class for predictors.
    """
    pass


class LLMPredictor(Predictor):
    """
    Base class for LLMPredictors that use Large Language Models (LLMs)
    to make sample predictions given text instructions:
    prediction = LLM(sample, instructions)
    """

    @abstractmethod
    def predict_row(self, row: pd.Series, instruction: str, labels: List[str]) -> str:
        """
        Predict a single row from a pandas DataFrame.
        To be used with pandas.DataFrame.apply:
        df.apply(func=predict_row, axis=1, instruction=instruction)
        """

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        instruction: str,
        labels: List[str],
        output_column: str = 'predictions'
    ) -> pd.DataFrame:
        """
        Predict all rows from a pandas DataFrame.
        """


class LangChainLLMPredictor(LLMPredictor):
    llm: Optional[BaseLLM] = None
    llm_chain: Optional[LLMChain] = None
    prompt_template: Optional[str] = None
    verbose: bool = False

    @root_validator
    def initialize_llm(cls, values):
        if values.get('prompt_template') is None:
            default_file = Path(__file__).parent / 'prompts' / 'simple_classification.txt'
            with open(default_file, 'r') as f:
                values['prompt_template'] = f.read()
        if values.get('llm') is None:
            values['llm'] = OpenAI(model_name='text-davinci-003', temperature=0)
        if values.get('llm_chain') is None:
            values['llm_chain'] = LLMChain(
                llm=values['llm'],
                prompt=values['prompt_template'],
                verbose=values['verbose']
            )
        return values

    def predict_row(self, row: pd.Series, instruction: str, labels: List[str]) -> str:
        row_dict = row.to_dict()
        prediction = self.llm_chain.predict(
            record=json.dumps(row_dict),
            instructions=instruction,
            labels=str(labels)
        )
        # match prediction to labels
        scores = list(map(lambda l: difflib.SequenceMatcher(None, prediction, l).ratio(), labels))
        safe_prediction = labels[scores.index(max(scores))]
        return safe_prediction

    def predict(
        self,
        df: pd.DataFrame,
        instruction: str,
        labels: List[str],
        output_column: str='predictions'
    ) -> pd.DataFrame:

        predictions = df.apply(
            func=self.predict_row,
            axis=1,
            instructions=instruction,
            labels=labels,
        )
        return df.assign(**{output_column: predictions})
