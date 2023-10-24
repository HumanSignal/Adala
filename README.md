<a href="#"><img src="/static/logo.png" alt="ADALA logo" width="275" ></a>

Adala is an **A**utonomous **DA**ta (**L**abeling) **A**gent framework.

Adala offers a robust framework for implementing agents specialized in data processing, with an emphasis on
diverse data labeling tasks. These agents are autonomous, meaning they can independently acquire one or more skills
through iterative learning. This learning process is influenced by their operating environment, observations, and
reflections. Users define the environment by providing a ground truth dataset. Every agent learns and applies its skills
in what we refer to as a "runtime", synonymous with LLM.

![Diagram of components](./static/diagram.png "Diagram of components")

<!-- Offered as an HTTP server, users can interact with Adala via command line or RESTful API, and directly integrate its features in Python Notebooks or scripts. The self-learning mechanism leverages Large Language Models (LLMs) from providers like OpenAI and VertexAI. -->

## 📢 Why choose Adala?

- 🌟 **Reliable agents**: Our agents are built upon a foundation of ground
  truth data. This ensures consistent and trustworthy results, making Adala a
  reliable choice for your data processing needs.
  
- 🎮 **Controllable output**: For every skill, you can configure the
  desired output and set specific constraints with varying degrees of
  flexibility. Whether you want strict adherence to particular
  guidelines or more adaptive outputs based on the agent's learning,
  Adala allows you to tailor results to your exact needs.

- 🎯 **Specialized in data processing**: While our agents excel in diverse
  data labeling tasks, they can be customized for a wide range of data
  processing needs.
  
- 🧠 **Autonomous learning**: Adala agents aren't just automated;
  they're intelligent. They iteratively and independently develop
  skills based on environment, observations, and reflections.

- ✅ **Flexible and extensible runtime**: Adala's runtime environment is
  adaptable. A single skill can be deployed across multiple runtimes,
  facilitating dynamic scenarios like the student/teacher
  architecture. Moreover, the openness of our framework invites the
  community to extend and tailor runtimes, ensuring continuous
  evolution and adaptability to diverse needs.
  
- 🚀 **Easily customizable**: Quickly customize and develop agents to address
  challenges specific to your needs, without facing a steep learning curve.

## 🫵 Who is Adala for?

Adala is a versatile framework designed for individuals and professionals in the field of AI and machine learning. Here's who can benefit:

- 🧡 **AI engineers:** Architect and design AI agent systems with modular, interconnected skills. Build production-level agent systems, abstracting low-level ML to Adala and LLMs.
- 💻 **Machine learning researchers:** Experiment with complex problem decomposition and causal reasoning.
- 📈 **Data scientists:** Apply agents to preprocess and postprocess your data. Interact with Adala natively through Python notebooks when working with large Dataframes.
- 🏫 **Educators and students:** Use Adala as a teaching tool or as a base for advanced projects and research.

While the roles highlighted above are central, it's pivotal to note that Adala is intricately designed to streamline and elevate the AI development journey, 
catering to all enthusiasts, irrespective of their specific niche in the field. 🥰

## ⬇️ Installation

Install Adala:

```sh
git clone https://github.com/HumanSignal/adala.git
cd adala/
pip install -e .
```

If you're planning to use human-in-the-loop labeling, or need a labeling tool to produce ground truth datasets, we
suggest installing Label Studio. Adala supports Label Studio format out of the box.

```sh
pip install label-studio
```

## 📝 Prerequisites

Set OPENAI_API_KEY ([see instructions here](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key))

```
export OPENAI_API_KEY='your-openai-api-key'
```

## 🎬 Quickstart

In this example we will use Adala as a standalone library directly inside our Python notebook. You can open it in Colab.

Click [here](./adala/examples/quickstart.ipynb) to see an extended quickstart example. 

```python
import pandas as pd

from adala.agents import Agent
from adala.datasets import DataFrameDataset
from adala.environments import BasicEnvironment
from adala.skills import ClassificationSkill
from rich import print

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

print("=> Initialize and train Adala agent ...")
agent = Agent(
    # connect to a dataset
    environment=BasicEnvironment(
        ground_truth_dataset=train_dataset,
        ground_truth_column="ground_truth"
    ),
    # define a skill
    skills=ClassificationSkill(
        name='sentiment_classification',
        instructions="Label text as subjective or objective.",
        labels=["Positive", "Negative", "Neutral"],
        input_data_field='text'
    ),
    
    # uncomment this if you want more quality and you have access to OPENAI GPT-4 model
    # default_teacher_runtime='openai-gpt4',
)
print(agent)

agent.learn(learning_iterations=3, accuracy_threshold=0.95)
print(agent.skills)

print('\n=> Run tests ...')
run = agent.apply_skills(test_dataset)
print('\n => Test results:')
print(run)
```

### Available skills
- [ClassificationSkill](./adala/examples/classification_skill.ipynb) – Classify text into a set of predefined labels.
- [ClassificationSkillWithCoT](./adala/examples/classification_skill_with_CoT.ipynb) – Classify text into a set of predefined labels, using Chain-of-Thoughts reasoning.
- [SummarizationSkill](./adala/examples/summarization_skill.ipynb) – Summarize text into a shorter text.
- [QuestionAnsweringSkill](./adala/examples/question_answering_skill.ipynb) – Answer questions based on a given context.
- [TranslationSkill](./adala/examples/translation_skill.ipynb) – Translate text from one language to another.
- [TextGenerationSkill](./adala/examples/text_generation_skill.ipynb) – Generate text based on a given prompt.
<!-- 
## 📒 More notebooks

- [Quickstart](./adala/examples/quickstart.ipynb) – An extended example of the above with comments and outputs.
- [Creating New Skill (coming soon!)](./adala/examples/creating_new_skill.ipynb) – An example that walks you through creating a new skill.
- [Label Studio Tutorial (coming soon!)](examples/tutorial_label_studio.ipynb) – An example of connecting Adala to an external labeling tool for enhanced supervision.
-->
<!-- 
## Running ADALA as a standalone server (Coming soon!)

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

## 🗺 Roadmap

- [ ] Create Named Entity Recognition Skill
- [ ] Extend environment with one more example
- [ ] Command line utility (see the source for this readme for example)
- [ ] REST API to interact with Adala
- [ ] Multi-task learning (learn multiple skills at once)
- [ ] Vision and multi-modal agent skills

## 🤩 Contributing to Adala

Enhance skills, optimize runtimes, or pioneer new agent types. Whether you're
crafting nuanced tasks, refining computational environments, or sculpting specialized agents for unique domains, your
contributions will power Adala's evolution. Join us in shaping the future of intelligent systems and making Adala more
versatile and impactful for users across the globe.

[Read more](./CONTRIBUTION.md) here.

## 💬 Support

Do you need help or are you looking to engage with our community? Check out our [Discord channel](https://discord.gg/QBtgTbXTgU)!
Whether you have questions, need clarification, or simply want to discuss topics related to our project, the Discord community is welcoming!
