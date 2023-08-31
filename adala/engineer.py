import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain


class Engineer:

    SYSTEM_MESSAGE = open(os.path.join(os.path.dirname(__file__), 'prompts', 'engineer_system_message.txt')).read()
    HUMAN_MESSAGE = open(os.path.join(os.path.dirname(__file__), 'prompts', 'engineer_human_message.txt')).read()

    def __init__(self):
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

    def __call__(self, current_instructions, observations):
        new_instructions = self.chain.run(
            instructions=current_instructions,
            observations=observations
        )
        return new_instructions
