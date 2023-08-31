import pandas as pd
import os

from typing import List, Dict
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import create_pandas_dataframe_agent


class Analyst:
    PROMPT_TEMPLATE = open(os.path.join(os.path.dirname(__file__), 'prompts', 'analyst.txt')).read()

    def __init__(self):
        self.llm = OpenAI(model_name='gpt-4', temperature=0)

    def __call__(self, df: pd.DataFrame):
        agent = create_pandas_dataframe_agent(llm=self.llm, df=df, verbose=True)
        explorer_prompt = self.PROMPT_TEMPLATE
        observations = agent.run(explorer_prompt)
        return observations
