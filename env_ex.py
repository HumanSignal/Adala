import pandas as pd
import adala import agents, skills, datasets, runtimes, envs

# this is the dataset we will use to train our agent
# filepath = "path/to/dataset.csv"
# df = pd.read_csv(filepath, sep='\t', nrows=100)

texts = [
    "The mic is great.", 
    "Will order from them again!",
    "Not loud enough and doesn't turn on like it should.",
    "The phone doesn't seem to accept anything except CBR mp3s",
    "All three broke within two months of use."
]

df = pd.DataFrame(texts, columns=['text'])

gt_texts = [["This is great", "Positive"], ["This is amazing", "Positive"]]
gt_df = pd.DataFrame(gt_texts, columns=['text', 'ground_truth'])

# for this example we will use OpenAI GPT4 model as a runtime
gpt4_runtime = runtimes.OpenAIRuntime(llm_params={
    'model': 'gpt-4',
})

env = envs.TerminalEnvironment(dataset=datasets.DataFrameDataset(df=gt_df))

agent = agents.SingleShotAgent(
    environment=env
    
    # initialize the runtime, you can have as many runtimes as you want
    runtimes={ "openai-gpt4": gpt4_runtime },
    
    # if you don't want to pass the environment to each skill just
    # provide it a default
    default_runtime="openai-gpt4",
    
    # define a skill
    # skill = skills.LabelingSkill(labels=['Positive', 'Negative', 'Neutral']),
    
    skills={ "classify": skills.ClassificationSkill(labels=['Positive', 'Negative', 'Neutral']) }
)

dataset = datasets.DataFrameDataset(df=df)
predictions = agent.predict(dataset)

print(predictions)

for _ in range(3):
    # agent learns and improves from the ground truth signal
    learnings = agent.learn(update_instructions=True)
    
    # display results
    print(learnings.experience.accuracy)


predictions = agent.predict(dataset)

low_conf=predictions[['confidence' < 10]]

# get feedback from the user
env.request_feedback(agent.get_skill("classify"), low_conf)

# do learnings again 
learnings = agent.learn(update_instructions=True)
    
# display results
print(learnings.experience.accuracy)


