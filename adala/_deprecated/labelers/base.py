import pandas as pd
import difflib
import json
import re
import openai
import guidance

from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from pydantic import BaseModel, root_validator
from typing import Optional, List, Dict
from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, PromptTemplate


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
    prediction_column: str = 'predictions'
    score_column: str = 'score'

    @abstractmethod
    def label_string(self, input_string: str, instruction: str, labels: List[str]) -> Dict:
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
        labels: List[str]
    ) -> pd.DataFrame:
        """
        Label all rows from a pandas DataFrame.
        """


class OpenAILabeler(LLMLabeler):
    model_name: str = 'gpt-3.5-turbo-instruct'
    temperature: float = 0
    prompt_template: str = '{{instruction}}\nInput: {{input}}\nOutput: {{select "output" options=labels logprobs="logprobs"}}'
    verbose: bool = False

    _llm = None

    @root_validator
    def initialize_llm(cls, values):
        values['_llm'] = guidance(
            template=values.get('prompt_template'),
            llm=guidance.llms.OpenAI(values.get('model_name')),
            silent=not values.get('verbose')
        )
        return values

    def label_string(self, input_string: str, instruction: str, labels: Optional[List[str]] = None) -> Dict:
        result = self._llm(input=input_string, instruction=instruction, labels=labels)
        return {
            self.prediction_column: result['output'],
            self.score_column: result['logprobs'][result['output']]
        }

    def label_row(self, row: pd.Series, instruction: str, labels: List[str]) -> Dict:
        return self.label_string(row.to_json(force_ascii=False), instruction, labels)

    def label(
        self,
        df: pd.DataFrame,
        instruction: str,
        labels: List[str],
        output_column: str = 'predictions'
    ) -> pd.DataFrame:
        tqdm.pandas(desc='Labeling')
        df[[self.prediction_column, self.score_column]] = df.progress_apply(
            func=self.label_row,
            axis=1,
            result_type='expand',
            instruction=instruction,
            labels=labels,
        )
        return df


class LangChainLabeler(LLMLabeler):
    model_name: str = 'gpt-3.5-turbo'
    temperature: float = 0
    llm: Optional[BaseLLM] = None
    llm_chain: Optional[LLMChain] = None
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    verbose: bool = False

    @root_validator
    def initialize_llm(cls, values):
        if values.get('prompt') is None:
            # default_file = Path(__file__).parent / 'prompts' / 'simple_classification.txt'
            # with open(default_file, 'r') as f:
            #     values['prompt'] = f.read()
            values['prompt'] = '{instructions}\n\nInput:\n{record}\n\nOutput:\n'
        if values.get('llm') is None:
            values['llm'] = ChatOpenAI(
                model_name=values['model_name'],
                temperature=values['temperature']
            )
        if values.get('llm_chain') is None:
            prompt = HumanMessagePromptTemplate(prompt=PromptTemplate.from_template(values['prompt']))
            messages = [prompt]
            if values.get('system_prompt') is not None:
                system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate.from_template(values['system_prompt']))
                messages.insert(0, system_prompt)
            values['llm_chain'] = LLMChain(
                llm=values['llm'],
                prompt=ChatPromptTemplate.from_messages(messages=messages),
                verbose=values['verbose']
            )
        return values

    def label_string(self, input_string: str, instruction: str, labels: Optional[List[str]] = None):
        prediction = self.llm_chain.run(
            record=input_string,
            instructions=instruction,
            # labels=str(labels)
        )
        if labels:
            prediction = prediction.strip()
            line_predictions = []
            # for line_prediction in prediction.split('\n'):
            for line_prediction in re.split(r',|\n', prediction):

                # match prediction to actual labels
                scores = list(map(lambda l: difflib.SequenceMatcher(None, line_prediction.strip(), l).ratio(), labels))
                line_prediction = labels[scores.index(max(scores))]
                line_predictions.append(line_prediction)
            prediction = ','.join(sorted(line_predictions))
        return prediction

    def label_row(self, row: pd.Series, instruction: str, labels: Optional[List[str]] = None) -> Dict:
        return self.label_string(input_string=row.to_json(force_ascii=False), instruction=instruction, labels=labels)

    def label(
        self,
        df: pd.DataFrame,
        instruction: str,
        labels: Optional[List[str]] = None,
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
