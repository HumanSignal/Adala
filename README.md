[![PyPI version](https://badge.fury.io/py/adala-pk-test.svg)](https://badge.fury.io/py/adala-pk-test)
![GitHub Repo stars](https://img.shields.io/github/stars/HumanSignal/Adala)
[![](https://img.shields.io/discord/1166330284300570624?label=Discord&logo=discord)](https://discord.gg/QBtgTbXTgU)

<a href="#"><img src="/static/logo.png" alt="ADALA logo" width="275" ></a>

Adala is an **A**utonomous **DA**ta (**L**abeling) **A**gent framework.

Adala offers a robust framework for implementing agents specialized in data processing, with a particular emphasis on
diverse data labeling tasks. These agents are autonomous, meaning they can independently acquire one or more skills
through iterative learning. This learning process is influenced by their operating environment, observations, and
reflections. Users define the environment by providing a ground truth dataset. Every agent learns and applies its skills
in what we refer to as a "runtime", synonymous with LLM.

![Diagram of components](./static/diagram.png "Diagram of components")

<!-- Offered as an HTTP server, users can interact with Adala via command line or RESTful API, and directly integrate its features in Python Notebooks or scripts. The self-learning mechanism leverages Large Language Models (LLMs) from providers like OpenAI and VertexAI. -->

### Why Choose Adala?

- **Reliable Agents**: Built upon a foundation of ground truth data,
  our agents ensure consistent and trustworthy results, making Adala a
  reliable choice for data processing needs.
  
- **Controllable Output**: For every skill, you can configure the
  desired output, setting specific constraints with varying degrees of
  flexibility. Whether you want strict adherence to particular
  guidelines or more adaptive outputs based on the agent's learning,
  Adala allows you to tailor results to your exact needs.

- **Specialized in Data Processing**: While our agents excel in diverse
  data labeling tasks, they can be tailored to a wide range of data
  processing needs.
  
- **Autonomous Learning**: Adala agents aren't just automated;
  they're intelligent. They iteratively and independently develop
  skills based on environment, observations, and reflections.

- **Flexible and Extensible Runtime**: Adala's runtime environment is
  adaptable. A single skill can be deployed across multiple runtimes,
  facilitating dynamic scenarios like the student/teacher
  architecture. Moreover, the openness of our framework invites the
  community to extend and tailor runtimes, ensuring continuous
  evolution and adaptability to diverse needs.
  
- **Extend Skills**: Quickly tailor and develop agents to address the
  specific challenges and nuances of your domain, without facing a
  steep learning curve.

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

from adala.agents import Agent
from adala.datasets import DataFrameDataset
from adala.environments import BasicEnvironment
from adala.skills import ClassificationSkill
from adala.utils.logs import print_evaluations

print("=> Initialize datasets ...")

# Train dataset
train_df = pd.DataFrame([
    ["It was the negative first impressions, and then it started working.", "Positive"],
    ["Not loud enough and doesn't turn on like it should.", "Negative"],
    ["I don't know what to say.", "Neutral"],
    ["Manager was rude, but the most important that mic shows very flat frequency response.", "Positive"],
    ["The phone doesn't seem to accept anything except CBR mp3s.", "Negative"],
    ["I tried it before, I bought this device for my son.", "Neutral"],
], columns=["text", "ground_truth"])

# Test dataset
test_df = pd.DataFrame([
    "All three broke within two months of use.",
    "The device worked for a long time, can't say anything bad.",
    "Just a random line of text.",
    "Will order from them again!",
], columns=["text"])

train_dataset = DataFrameDataset(df=train_df)
test_dataset = DataFrameDataset(df=test_df)

print("=> Initialize and train ADALA agent ...")
agent = Agent(
    # connect to a dataset
    environment=BasicEnvironment(
        ground_truth_dataset=train_dataset,
        ground_truth_column="ground_truth"
    ),
    # define a skill
    skills=ClassificationSkill(
        name='sentiment',
        instructions="Label text as subjective or objective.",
        labels=["Positive", "Negative", "Neutral"],
        input_data_field='text'
    ),
)
run = agent.learn(learning_iterations=10, accuracy_threshold=0.95)

print('\n\n=> Final instructions:')
print('=====================')
print(f'{run.updated_instructions}')
print('=====================')

print('\n=> Run test ...')
run = agent.apply_skills(test_dataset)
print_evaluations(run.predictions)
```

## More Notebooks

- [Quickstart](./adala/examples/quickstart.ipynb) – An extended example of the above with comments and outputs.
- [Creating New Skill](./adala/examples/creating_new_skill.ipynb) – An example that walks you through creating a new skill.
- [Label Studio Tutorial](./adala/examples/tutorial_label_studio.ipynb) – An example of connecting Adala to an external labeling tool for enhanced supervision.

<!-- 
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
-->

### Who Adala is for?

Adala is a versatile framework designed for individuals and professionals in the field of AI and machine learning. Here's who can benefit:

- AI Engineers - Architect and design AI Agent systems with modular, interconnected skills. Build production-level agent systems, abstracting low-level ML to Adala and LLMs.
- Machine Learning Researchers - Experiment with complex problem decomposition and causal reasoning.
- Data Scientists - Apply agents to preprocess and postprocess your data. Interact with Adala natively through Python notebooks when working with large Dataframes.
- Educators and Students - Use Adala as a teaching tool or as a base for advanced projects and research.

While the roles highlighted above are central, it's pivotal to note that Adala is intricately designed to streamline and elevate the AI development journey, catering to all enthusiasts, irrespective of their specific niche in the field.

## Roadmap

- [ ] Create Named Entity Recognition Skill
- [ ] Extend Environemnt with one more example
- [ ] Command Line Utility (see the source for this readme for example)
- [ ] REST API to interact with Adala

## Contributing to Adala

Dive into the heart of Adala by enhancing Skills, optimizing Runtimes, or pioneering new Agent Types. Whether you're
crafting nuanced tasks, refining computational environments, or sculpting specialized agents for unique domains, your
contributions will power Adala's evolution. Join us in shaping the future of intelligent systems and making Adala more
versatile and impactful for users across the globe.

[Read more](./CONTRIBUTION.md) here.

## Support

Are you in need of assistance or looking to engage with our community? Our [Discord channel](https://discord.gg/QBtgTbXTgU)) is the perfect place for real-time support and interaction. Whether you have questions, need clarifications, or simply want to discuss topics related to our project, the Discord community is welcoming!

## FAQ

- What is an agent?
  - Agent is a set of skills and runtimes that could be used to execute those skills. Each agent has its own unique environment (dataset) attached to it. You can define your own agent class that would have a unique set of skills for your domain.
- What is a skill?
  - Skill is a learned ability to solve a specific task. Skill gets trained from the ground truth dataset.

