import pandas as pd

from adala.agents import SingleShotAgent
from adala.runtimes import LLMRuntime, LLMRuntimeModelType
from adala.datasets import DataFrameDataset, LabelStudioDataset
from adala.skills import ClassificationSkill, ClassificationSkillWithReasoning, TextGenerationSkill
from adala.memories import FileMemory

# 1. Connect to Dataset
# we can use pandas dataframe as a dataset...
filepath = 'https://hs-sandbox-pub.s3.amazonaws.com/amazon_cells_labelled.tsv'
pandas_dataset = DataFrameDataset(
    df=pd.read_csv(filepath, sep='\t', nrows=100)
)

# ... or use Label Studio project to dynamically load data from a remote server
label_studio_dataset = LabelStudioDataset(
    url='http://localhost:8080',
    api_key='1234567890',
    project_id=1
)

# 2. Create a target skill

# if we have a classification task, we can use ClassificationSkill
classification_skill = ClassificationSkill(
    name='subjectivity_detection',
    description='Understanding subjective and objective statements from text.',
    instructions='Classify a product review as either expressing "Subjective" or "Objective" statements.'
)

# alternatively, we can use TextGenerationSkill to generate a free-form text instead
text_generation_skill = TextGenerationSkill(
    name='opinion',
    description='Provide your opinion on the product.',
    instructions='Write a product review.'
)


# 3. Create an agent
agent = SingleShotAgent(

    # use dataset
    dataset=pandas_dataset,

    # create runtime
    runtime=LLMRuntime(
        llm_runtime_type=LLMRuntimeModelType.OpenAI
    ),

    # add agent memory
    memory=FileMemory(
        filepath='long_term_memory.jsonl'
    ),

    # enable skill
    skill=classification_skill,

    # we can also combine multiple skills
    # skillset=[text_generation_skill, classification_skill]
)

step_result = agent.step()

while not step_result.is_last:
    print(step_result.summarize())
    step_result = agent.step()

print('Done!')
print(agent.report())
