# ADALA

Adala is an Autonomous DatA (Labeling) Agent framework.

Adala offers a robust framework for implementing agents specialized in data processing, with a particular emphasis on
diverse data labeling tasks. These agents are autonomous, meaning they can independently acquire one or more skills
through iterative learning. This learning process is influenced by their operating environment, observations, and
reflections. Users define the environment by providing a ground truth dataset. Every agent learns and applies its skills
in what we refer to as a "runtime", synonymous with LLM.

Offered as an HTTP server, users can interact with Adala via command line or RESTful API, and directly integrate its
features in Python Notebooks or scripts. The self-learning mechanism leverages Large Language Models (LLMs) from
providers like OpenAI and VertexAI.

### Why Choose Adala?

- **Specialized in Data Processing**: While our agents excel in diverse data labeling tasks, they can be tailored to a
  wide range of data processing needs.
- **Autonomous Learning**: Adala agents aren't just automated; they're intelligent. They iteratively and independently
  develop skills based on environment, observations, and reflections.
- **User-Centric Environment Setup**: You have control. Define your agent's learning environment simply by providing a
  ground truth dataset.
- **Optimized Runtime**: Our agents operate in a state-of-the-art runtime environment, synonymous with LLM, ensuring
  efficiency and adaptability.
- **Extend to your domain**: Build custom agents and skills focused on your specific domain.

## Installation

Install ADALA:

```sh
git clone https://github.com/HumanSignal/ADALA.git
cd ADALA/
pip install -e .
```

If you're planning to use human-in-the-loop labeling, or need a labeling tool to produce ground truth datasets, we
suggest installing Label Studio. Adala is made to support Label Studio format right out of the box.

```sh
pip install label-studio
```

## Prerequisites

Set OPENAI_API_KEY ([see instructions here](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key))

```
export OPENAI_API_KEY='your-openai-api-key'
```

## Quickstart

In this example we will use ADALA as a standalone library directly inside our python notebook. You can open it in Collab
right here.

```python
import pandas as pd
import textwrap as tw

from adala.agents import SingleShotAgent
from adala.datasets import DataFrameDataset
from adala.skills import ClassificationSkill

# this is the dataset we will use to train our agent
# filepath = "path/to/dataset.csv"
# df = pd.read_csv(filepath, sep='\t', nrows=100)

print('\n=> Prepare dataset and ADALA agent')
texts = [
    "It was the negative first impressions, and then it started working.",
    "Not loud enough and doesn't turn on like it should.",
    "I don't know what to say.",
    "Manager was rude, but the most important that mic shows very flat frequency response.",
    "The phone doesn't seem to accept anything except CBR mp3s.",
    "I tried it before, I bought this device for my son.",
    "All three broke within two months of use.",
    "The device worked for a long time, can't say anything bad.",
    "Just a random line of text.",
    "Will order from them again!",
]
df = pd.DataFrame(texts, columns=['text'])

agent = SingleShotAgent(
    # connect to a dataset
    dataset=DataFrameDataset(df=df),
    # define a skill
    skill = ClassificationSkill(labels=['Positive', 'Negative', 'Neutral']),
)

print('\n=> Agent run')
run = agent.run()
# display results
result = pd.concat((df, run.experience.predictions), axis=1)
print('Agent results without training:\n', result[["text", "ground_truth", "predictions"]])

# provide ground truth signal in the original dataset
df.loc[0, 'ground_truth'] = 'Positive'
df.loc[1, 'ground_truth'] = 'Negative'
df.loc[2, 'ground_truth'] = 'Neutral'
df.loc[3, 'ground_truth'] = 'Positive'
df.loc[4, 'ground_truth'] = 'Negative'
df.loc[5, 'ground_truth'] = 'Neutral'
# df.loc[10, 'ground_truth'] = 'None'

print('\n=> Train agent\n')
for i in range(10):
    print(f'===> Iteration {i+1}:')
    # agent learns and improves from the ground truth signal
    learnings = agent.learn(update_instructions=True)
    text = learnings.experience.updated_instructions
    text = tw.fill(text, width=100, initial_indent=" "*4, subsequent_indent=" "*4)
    table = pd.concat((df, learnings.experience.predictions), axis=1)

    # display results
    print(f'  accuracy = {learnings.experience.accuracy}')
    print(f'  updated instructions = \n{text}')
    print(f'  results =\n{table[["text", "ground_truth", "predictions"]]}\n')

    if learnings.experience.accuracy >= 1.0:
        break
```

Check [more examples in notebook tutorials.](https://github.com/HumanSignal/ADALA/tree/master/adala/examples)

## Running ADALA as a standalone server (Comming soon!)

Initiate the Adala server. Note: Each agent operates as its own web server.

### Starting the Adala Server

```sh
# Start the Adala server on default port 8090
adala start
```

### Uploading Ground Truth Data

Before teaching skills to Adala, you need to set up the environment and upload data.

```sh
# Upload your dataset
adala upload --file sample_dataset_ground_truth.json
```

### Teaching Skills to Adala

Now, define and teach a new skill to Adala.

```sh
# Define a new skill for classifying objects
adala add-skill --name "Object Classification" --description "Classify text into categories." --instruction "Example: Label trees, cars, and buildings."
```

```sh
# Start the learning process
adala learn --skill "Object Classification" --continuous
```

### Monitoring Optimization

Track the progress of the optimization process.

```sh
# Check the optimization status
adala status
```

### Applying Skills and Predictions

You don't need to wait for optimization to finish. Instruct Adala to apply its skills on new data outside the
environment, turning Adala into a prediction engine. If the predictions generated by the skill are then verified by
human validators or another supervision system, this provides more ground truth data, enhancing the agent's skills. Use
the learned skills and generate predictions.

```sh
# Apply the 'Object Classification' skill on new data
adala apply-skill --name "Object Classification" --file sample_dataset_predict.json
```

### Review Metrics

Get insights into Adala's performance.

```sh
# View detailed metrics
adala metrics
```

## Executing ADALA Command Line

```sh
# Start the Adala server on default port 8090
adala start --port 8090

# Upload your dataset
adala upload --file sample_dataset_ground_truth.json

# Define a new skill for classifying objects
adala add-skill --name "Object Classification" --description "Classify images into categories." --instruction "Example: Label trees, cars, and buildings."

# Start the learning process
adala learn --skill "Object Classification"

# Check the optimization status
adala status

# Apply the 'Object Classification' skill on new data
adala apply-skill --name "Object Classification" --file sample_dataset_predict.json

# View detailed metrics
adala metrics

# Restart the Adala server
adala restart

# Shut down the Adala server
adala shutdown

# List all the skills
adala list-skills

# List all the runtimes
adala list-runtimes

# Retrieve raw logs
adala logs

# Provide help
adala help <command>
```

## Contributing to ADALA

Dive into the heart of Adala by enhancing our Skills, optimizing Runtimes, or pioneering new Agent Types. Whether you're
crafting nuanced tasks, refining computational environments, or sculpting specialized agents for unique domains, your
contributions will power Adala's evolution. Join us in shaping the future of intelligent systems and making Adala more
versatile and impactful for users across the globe.

Read more here.

## How ADALA compares to other agent libraries

## FAQ

- What is an agent?
- Agent is a set of skills and runtimes that could be used to execute those skills. Each agent has its own unique
  environment (dataset)
  attached to it. You can define your own agent class that would have a unique set of skills for your domain.

-

## Interesting Stuff

Skill is a learned ability to solve a specific task. Skill gets trained from the ground truth dataset. 
