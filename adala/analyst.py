import pandas as pd
import os

from typing import List, Dict
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory


class Analyst:
    PROMPT_TEMPLATE = open(os.path.join(os.path.dirname(__file__), 'prompts', 'analyst_2.txt')).read()

    def __call__(self, df: pd.DataFrame):
        llm = OpenAI(model_name='gpt-4', temperature=0)
        memory = ConversationBufferMemory(memory_key='history')
        df = df.copy()
        df = df[df['ground_truth'] != df['predictions']]
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            memory=memory
        )
        explorer_prompt = self.PROMPT_TEMPLATE
        observations = agent.run(explorer_prompt)
        return observations
