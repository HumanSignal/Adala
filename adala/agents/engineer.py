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


class Engineer2:
    SYSTEM_MESSAGE = '''\
Act as an 'Instruction Tuner' for the LLM. You will be given two primary inputs:

- The [CURRENT INSTRUCTION] used to guide the LLM's classification
- Target set of [LABELS] for the dataset in question.
- [CURRENT ERRORS] that emerged when this instruction was applied to a dataset.

The ERRORS presented in JSON format, which contain the ground_truth label, \
the predictions label, and the input data in one or more columns. \
Here's an example format for the errors:

```json
[{{"ground_truth":"...","predictions":"...", "input_text": "...", "other": "data", ...}}, {{"ground_truth":"...","predictions":"...", "input_text": "...", "other": "data", ...}}, ...]
```

Analyze these inputs and craft a revised instruction for the LLM, aiming to enhance classification accuracy for the dataset in question. Deliver your response as the refined instruction.
'''
#     SYSTEM_MESSAGE = '''\
# Act as an 'Instruction Tuner' for the LLM. \
# You will be presented with information from two previous rounds of instructions, \
# as well as associated errors presented in JSON format, which contain the ground_truth label, \
# the predictions label, and the input data in one or more columns. \
# Here's an example format for the errors:
#
# ```json
# [{{"ground_truth":"...","predictions":"...", "input_text": "...", "other": "data", ...}}, {{"ground_truth":"...","predictions":"...", "input_text": "...", "other": "data", ...}}, ...]
# ```
#
# Here's an example format for the instructions:
#
# Instruction from Round 1: [previous instruction 1]
# Error observed from Round 1: [JSON format error data from previous instruction 1]
# Instruction from Round 2: [previous instruction 2]
# Error observed from Round 2: [JSON format error data from previous instruction 2]
#
# Expected output labels: [LABELS]
#
# Your task is to deeply analyze the provided instructions alongside their respective errors. \
# After understanding the discrepancies, craft a new instruction for the LLM. \
# This instruction should aim to reduce the observed errors from both rounds. \
# Provide your refined instruction as the response.
# '''
    # HUMAN_MESSAGE = open(os.path.join(os.path.dirname(__file__), 'prompts', 'engineer_human_message_2.txt')).read()
    HUMAN_MESSAGE = '''\
CURRENT INSTRUCTION: {instruction}
LABELS: {labels}
CURRENT ERRORS: {errors}

New refined instruction:
'''

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
            prompt=self.chat_prompt,
            # verbose=True
        )

    def __call__(self, instruction, errors, labels):
        new_instructions = self.chain.run(
            instruction=instruction,
            errors=errors,
            labels=labels
        )
        return new_instructions