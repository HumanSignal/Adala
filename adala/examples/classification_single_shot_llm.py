from adala.datasets import PandasDataframe
from adala.skills import LLMSkill
from adala.agents import SingleShotAgent

# example analogous to https://colab.research.google.com/drive/1Bn11v5X85PEgWnn3Rz_4GJRIUh3S0aEB

# create dataset
filepath = 'https://hs-sandbox-pub.s3.amazonaws.com/amazon_cells_labelled.tsv'
dataset = PandasDataframe(filepath, sep='\t', nrows=100)

# enable target skill
skill = LLMSkill('Classify a product review as either expressing "Subjective" or "Objective" statements.')

# create agent to improve the skill
agent = SingleShotAgent(dataset, skill)

# run agent
step_result = agent.step()
