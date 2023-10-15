import pandas as pd

from adala.agents import SingleShotAgent
from adala.runtimes import OpenAIGPTRuntime
from adala.datasets import PandasDataFrame
from adala.skills import LLMSkill
from adala.memories import FileMemory

filepath = 'https://hs-sandbox-pub.s3.amazonaws.com/amazon_cells_labelled.tsv'

# Create a new agent with a single skill
agent = SingleShotAgent(

    # use dataset
    dataset=PandasDataFrame(
        df=pd.read_csv(filepath, sep='\t', nrows=100)
    ),

    # create runtime
    runtime=OpenAIGPTRuntime(
        model_name='gpt-3.5-turbo-instruct'
    ),

    # add agent memory
    memory=FileMemory(
        filepath='long_term_memory.jsonl'
    ),

    # enable skill
    skill=LLMSkill(
        name='subjectivity_detection',
        description='Understanding subjective and objective statements from text.',
        instructions='Classify a product review as either expressing "Subjective" or "Objective" statements.'
    )
)

# make single step
step_result = agent.step()

# check new skill
print(agent.skill.instructions)

# check accuracy
print(step_result.artifact.experience.accuracy)
