import pandas as pd
import os

from typing import List, Dict
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain


class Analyst:
    SYSTEM_MESSAGE = '''\
You are tasked as the AI guide for a team of labelers at a company specializing in text classification. \
You'll be presented with a set of misclassified records, where the 'ground_truth' label differs \
from the 'predictions'. The data is formatted in JSON as follows:
[
{{"ground_truth": "ground truth label", "predictions": "predicted label", ...}},
...
]

Your objective is to closely examine these discrepancies, identify recurrent error patterns, \
and provide specific guidance to the labelers on how to rectify and avoid these mistakes \
in future labeling sessions.'''
    HUMAN_MESSAGE = '''\
ERRORS:
{errors}
GUIDANCE:
'''

    # PROMPT_TEMPLATE = open(os.path.join(os.path.dirname(__file__), 'prompts', 'analyst_2.txt')).read()

    def __call__(self, df: pd.DataFrame):
        self.llm = ChatOpenAI(model_name='gpt-4', temperature=0.5)
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_MESSAGE)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.HUMAN_MESSAGE)
        self.chat_prompt = ChatPromptTemplate.from_messages([
            self.system_message_prompt,
            self.human_message_prompt
        ])
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.chat_prompt
        )
        num_examples = 5
        df = df[df['ground_truth'] != df['predictions']].sample(num_examples)
        errors = df.to_json(orient='records')
        observations = self.chain.run(errors=errors)
        print(observations)
        return observations

        # llm = OpenAI(model_name='gpt-4', temperature=0)
        # memory = ConversationBufferMemory(memory_key='history')
        # df = df.copy()
        # df = df[df['ground_truth'] != df['predictions']]
        # agent = create_pandas_dataframe_agent(
        #     llm=llm,
        #     df=df,
        #     verbose=True,
        #     agent_executor_kwargs={'memory': memory},
        # )
        # explorer_prompt = self.PROMPT_TEMPLATE
        # observations = agent.run(explorer_prompt)
        # return observations
