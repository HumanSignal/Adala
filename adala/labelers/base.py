import pandas as pd
import difflib
import json

from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from pydantic import BaseModel, root_validator
from typing import Optional, List, Dict
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate


class Labeler(BaseModel, ABC):
    """
    Base class for labelers.
    """
    pass


class LLMLabeler(Labeler):
    """
    Base class for LLMLabeler that use Large Language Models (LLMs)
    to generate label predictions given text instructions:
    label = LLM(sample, instructions)
    """

    @abstractmethod
    def label_string(self, input_string: str, instruction: str, labels: List[str]) -> str:
        """
        Label a string with LLM given instruction:
        label = LLM(input_string, instruction)
        """

    @abstractmethod
    def label_row(self, row: pd.Series, instruction: str, labels: List[str]) -> str:
        """
        Label a single row from a pandas DataFrame.
        To be used with pandas.DataFrame.apply:
        df.apply(func=label_row, axis=1, instruction="Select 'A' or 'B'", labels=['A', 'B'])
        """

    @abstractmethod
    def label(
        self,
        df: pd.DataFrame,
        instruction: str,
        labels: List[str],
        output_column: str = 'predictions'
    ) -> pd.DataFrame:
        """
        Label all rows from a pandas DataFrame.
        """


class LangChainLabeler(LLMLabeler):
    model_name: str = 'gpt-3.5-turbo'
    temperature: float = 0
    llm: Optional[BaseLLM] = None
    llm_chain: Optional[LLMChain] = None
    prompt: Optional[str] = None
    verbose: bool = False

    @root_validator
    def initialize_llm(cls, values):
        if values.get('prompt') is None:
            default_file = Path(__file__).parent / 'prompts' / 'simple_classification.txt'
            with open(default_file, 'r') as f:
                values['prompt'] = f.read()
        if values.get('llm') is None:
            values['llm'] = ChatOpenAI(
                model_name=values['model_name'],
                temperature=values['temperature']
            )
        if values.get('llm_chain') is None:
            prompt = HumanMessagePromptTemplate(prompt=PromptTemplate.from_template(values['prompt']))
            values['llm_chain'] = LLMChain(
                llm=values['llm'],
                prompt=ChatPromptTemplate.from_messages([prompt]),
                verbose=values['verbose']
            )
        return values

    def label_string(self, input_string: str, instruction: str, labels: List[str]):
        prediction = self.llm_chain.run(
            record=input_string,
            instructions=instruction,
            labels=str(labels)
        )
        # match prediction to actual labels
        scores = list(map(lambda l: difflib.SequenceMatcher(None, prediction, l).ratio(), labels))
        return labels[scores.index(max(scores))]

    def label_row(self, row: pd.Series, instruction: str, labels: List[str]) -> str:
        return self.label_string(input_string=row.to_json(), instruction=instruction, labels=labels)

    def label(
        self,
        df: pd.DataFrame,
        instruction: str,
        labels: List[str],
        output_column: str = 'predictions'
    ) -> pd.DataFrame:

        tqdm.pandas(desc='Labeling')
        predictions = df.progress_apply(
            func=self.label_row,
            axis=1,
            instruction=instruction,
            labels=labels,
        )
        return df.assign(**{output_column: predictions})


class OpenAILabeler(LangChainLabeler):
    model_name = 'gpt-3.5-turbo'
