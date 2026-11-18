[![PyPI version](https://badge.fury.io/py/adala.svg)](https://badge.fury.io/py/adala)
![Python Version](https://img.shields.io/badge/supported_python_version_-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![GitHub](https://img.shields.io/github/license/HumanSignal/Adala)
![GitHub Repo stars](https://img.shields.io/github/stars/HumanSignal/Adala)
[![](https://img.shields.io/discord/1166330284300570624?label=Discord&logo=discord)](https://discord.gg/QBtgTbXTgU)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/docs/src/img/logo-dark-mode.png" width="275" >
  <source media="(prefers-color-scheme: light)" srcset="/docs/src/img/logo.png" width="275" >
  <img alt="Shows Adala logo in light mode and dark mode." src="/docs/src/img/logo.png" width="275" >
</picture>

Adala is an **A**utonomous **DA**ta (**L**abeling) **A**gent framework.

Adala offers a robust framework for implementing agents specialized in data processing, with an emphasis on
diverse data labeling tasks. These agents are autonomous, meaning they can independently acquire one or more skills
through iterative learning. This learning process is influenced by their operating environment, observations, and
reflections. Users define the environment by providing a ground truth dataset. Every agent learns and applies its skills
in what we refer to as a "runtime", synonymous with LLM.

![Training Agent Skill](./docs/src/img/training-agents-skill.png "Training Agent Skill")

<!-- Offered as an HTTP server, users can interact with Adala via command line or RESTful API, and directly integrate its features in Python Notebooks or scripts. The self-learning mechanism leverages Large Language Models (LLMs) from providers like OpenAI and VertexAI. -->

## 📢 Why choose Adala?

- 🌟 **Reliable agents**: Agents are built upon a foundation of ground
  truth data. This ensures consistent and trustworthy results, making Adala a
  reliable choice for your data processing needs.
  
- 🎮 **Controllable output**: For every skill, you can configure the
  desired output and set specific constraints with varying degrees of
  flexibility. Whether you want strict adherence to particular
  guidelines or more adaptive outputs based on the agent's learning,
  Adala allows you to tailor results to your exact needs.

- 🎯 **Specialized in data processing**: While agents excel in diverse
  data labeling tasks, they can be customized for a wide range of data
  processing needs.
  
- 🧠 **Autonomous learning**: Adala agents aren't just automated;
  they're intelligent. They iteratively and independently develop
  skills based on environment, observations, and reflections.

- ✅ **Flexible and extensible runtime**: Adala's runtime environment is
  adaptable. A single skill can be deployed across multiple runtimes,
  facilitating dynamic scenarios like the student/teacher
  architecture. Moreover, the openness of framework invites the
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

## 🔌Installation

Install Adala:

```sh
pip install adala
```

Adala frequently releases updates. In order to ensure that you are using the most up-to-date version, it is recommended that you install it from GitHub:

```sh
pip install git+https://github.com/HumanSignal/Adala.git
```

Developer installation:

```sh
git clone https://github.com/HumanSignal/Adala.git
cd Adala/
poetry install
```

<!--
If you're planning to use human-in-the-loop labeling, or need a labeling tool to produce ground truth datasets, we
suggest installing Label Studio. Adala supports Label Studio format out of the box.

```sh
pip install label-studio
```
-->

## 📝 Prerequisites

Set OPENAI_API_KEY ([see instructions here](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key))

```
export OPENAI_API_KEY='your-openai-api-key'
```

## 🎬 Quickstart

In this example we will use Adala as a standalone library directly inside Python notebook.

Click [here](./examples/quickstart.ipynb) to see an extended quickstart example.

```python
import pandas as pd

from adala.agents import Agent
from adala.environments import StaticEnvironment
from adala.skills import ClassificationSkill
from adala.runtimes import OpenAIChatRuntime
from rich import print

# Train dataset
train_df = pd.DataFrame([
    ["It was the negative first impressions, and then it started working.", "Positive"],
    ["Not loud enough and doesn't turn on like it should.", "Negative"],
    ["I don't know what to say.", "Neutral"],
    ["Manager was rude, but the most important that mic shows very flat frequency response.", "Positive"],
    ["The phone doesn't seem to accept anything except CBR mp3s.", "Negative"],
    ["I tried it before, I bought this device for my son.", "Neutral"],
], columns=["text", "sentiment"])

# Test dataset
test_df = pd.DataFrame([
    "All three broke within two months of use.",
    "The device worked for a long time, can't say anything bad.",
    "Just a random line of text."
], columns=["text"])

agent = Agent(
    # connect to a dataset
    environment=StaticEnvironment(df=train_df),

    # define a skill
    skills=ClassificationSkill(
        name='sentiment',
        instructions="Label text as positive, negative or neutral.",
        labels=["Positive", "Negative", "Neutral"],
        input_template="Text: {text}",
        output_template="Sentiment: {sentiment}"
    ),

    # define all the different runtimes your skills may use
    runtimes = {
        # You can specify your OPENAI API KEY here via `OpenAIRuntime(..., api_key='your-api-key')`
        'openai': OpenAIChatRuntime(model='gpt-4o'),
    },
    teacher_runtimes = {
        # You can specify your OPENAI API KEY here via `OpenAIRuntime(..., api_key='your-api-key')`
        'default': OpenAIChatRuntime(model='gpt-4o'),
    },
    default_runtime='openai',
)

print(agent)
print(agent.skills)

agent.learn(learning_iterations=3, accuracy_threshold=0.95)

print('\n=> Run tests ...')
predictions = agent.run(test_df)
print('\n => Test results:')
print(predictions)
```

However, if you prefer to use Adala with Claude, Gemini, or other OpenAI compatible LLMs, you can do so by using OpenRouter.ai. Below is an example of how to use the OpenRouter API:

Start by setting the `OPENROUTER_API_KEY` environment variable, which you can get from [OpenRouter](https://openrouter.ai/api-keys).

```
export OPENROUTER_API_KEY='your-openrouter-api-key'
```

Then, let's see how to modify the previous example to use OpenRouter with Claude 3.5 Haiku.

```python
import os
import pandas as pd

from adala.agents import Agent
from adala.environments import StaticEnvironment
from adala.skills import ClassificationSkill
from adala.runtimes import OpenAIChatRuntime
from rich import print

# Train dataset
train_df = pd.DataFrame([
    ["It was the negative first impressions, and then it started working.", "Positive"],
    ["Not loud enough and doesn't turn on like it should.", "Negative"],
    ["I don't know what to say.", "Neutral"],
    ["Manager was rude, but the most important that mic shows very flat frequency response.", "Positive"],
    ["The phone doesn't seem to accept anything except CBR mp3s.", "Negative"],
    ["I tried it before, I bought this device for my son.", "Neutral"],
], columns=["text", "sentiment"])

# Test dataset
test_df = pd.DataFrame([
    "All three broke within two months of use.",
    "The device worked for a long time, can't say anything bad.",
    "Just a random line of text."
], columns=["text"])

agent = Agent(
    # connect to a dataset
    environment=StaticEnvironment(df=train_df),

    # define a skill
    skills=ClassificationSkill(
        name='sentiment',
        instructions="Label text as positive, negative or neutral.",
        labels=["Positive", "Negative", "Neutral"],
        input_template="Text: {text}",
        output_template="Sentiment: {sentiment}"
    ),

    # define all the different runtimes your skills may use
    runtimes = {
        # You can specify your OpenRouter API Key here or set it ahead of time in your environment variable, OPENROUTER_API_KEY
        'openrouter': OpenAIChatRuntime(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-haiku",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            provider="Custom"
            ),
    },

    default_runtime='openrouter',

    teacher_runtimes = {
        "default" : OpenAIChatRuntime(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-haiku",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            provider="Custom"
        ),
    }
)

print(agent)
print(agent.skills)

agent.learn(learning_iterations=3, accuracy_threshold=0.95)

print('\n=> Run tests ...')
predictions = agent.run(test_df)
print('\n => Test results:')
print(predictions)
```

### 👉 Examples

| Skill                                                                              | Description                                                                       | Colab                                                                                                                                                                                                                                        |
|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ClassificationSkill](./examples/classification_skill.ipynb)                 | Classify text into a set of predefined labels.                                    | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/classification_skill.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>          |
| [ClassificationSkillWithCoT](./examples/classification_skill_with_CoT.ipynb) | Classify text into a set of predefined labels, using Chain-of-Thoughts reasoning. | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/classification_skill_with_CoT.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [SummarizationSkill](./examples/summarization_skill.ipynb)                   | Summarize text into a shorter text.                                               | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/summarization_skill.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |
| [QuestionAnsweringSkill](./examples/question_answering_skill.ipynb)          | Answer questions based on a given context.                                        | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/question_answering_skill.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>      |
| [TranslationSkill](./examples/translation_skill.ipynb)                       | Translate text from one language to another.                                      | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/translation_skill.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>             |
| [TextGenerationSkill](./examples/text_generation_skill.ipynb)                | Generate text based on a given prompt.                                            | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/text_generation_skill.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>         |
| [Skill sets](./examples/skillsets_sequence_of_skills.ipynb)                  | Process complex tasks through a sequence of skills.                               | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/skillsets_sequence_of_skills.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  |
| [OntologyCreator](./examples/ontology_creator.ipynb)                         | Infer ontology from a set of text examples.                                       | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/ontology_creator.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              |
| [Math reasoning](./examples/gsm8k_test.ipynb)                                 | Solve grade-school math problems on GSM8k dataset.                                | <a target="_blank" href="https://colab.research.google.com/github/HumanSignal/Adala/blob/master/examples/gsm8k_test.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                    |

![Executing Agent Skill](./docs/src/img/executing-agents-skill.png "Executing Agent Skill")

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

## ❓ FAQ

### General

**What is Adala?**
Adala (Autonomous DAta Labeling Agent) is a Python framework for building autonomous agents specialized in data processing tasks, with emphasis on diverse data labeling tasks. Agents independently acquire skills through iterative learning influenced by environment, observations, and reflections.

**How does Adala differ from other agent frameworks?**
Unlike general-purpose agent frameworks, Adala is specialized for data processing:
- **Reliable foundation**: Built on ground truth data for consistent results
- **Controllable output**: Configure desired output and constraints per skill
- **Autonomous learning**: Agents iteratively develop skills without manual intervention
- **Flexible runtime**: Deploy skills across multiple LLM runtimes (student/teacher architecture)

**What types of tasks is Adala best suited for?**
Adala excels at data labeling, classification, sentiment analysis, entity extraction, and other structured data processing tasks where ground truth datasets are available for agent learning.

### Installation & Setup

**What Python version is required?**
Python 3.8, 3.9, 3.10, or 3.11 are supported. Install via `pip install adala`.

**Which LLM providers are supported?**
- OpenAI (GPT-4o, GPT-3.5-turbo, etc.)
- Anthropic (via OpenRouter)
- VertexAI
- Any OpenAI-compatible API (OpenRouter, local servers)

Set the appropriate API key (`OPENAI_API_KEY` or `OPENROUTER_API_KEY`) before running.

**Can I use local models?**
Yes. Point the runtime to your local OpenAI-compatible server (Ollama, vLLM, etc.) via the `base_url` parameter in `OpenAIChatRuntime`.

### Core Concepts

**What is a "Skill" in Adala?**
A Skill defines a specific capability the agent performs, such as classification or entity extraction. Skills have:
- `name`: Skill identifier
- `instructions`: Task description
- `input_template`: How input data is formatted
- `output_template`: How output is formatted

**What is a "Runtime"?**
A Runtime is the LLM execution environment. Adala supports multiple runtimes per skill, enabling scenarios like student/teacher architecture where one LLM learns and another evaluates.

**What is an "Environment"?**
An Environment provides the ground truth dataset for learning. `StaticEnvironment` wraps a pandas DataFrame containing labeled examples.

**How does agent learning work?**
`agent.learn(learning_iterations=3, accuracy_threshold=0.95)` runs iterative learning:
1. Agent applies skill to training data
2. Runtime evaluates predictions against ground truth
3. Agent reflects on errors and adjusts approach
4. Process repeats until accuracy threshold is met

### Development

**How do I create a custom Skill?**
Extend base skill classes or create new ones with custom `instructions`, `input_template`, and `output_template`. See examples in `./examples/` directory.

**Can I use Adala in production?**
Yes. Adala is designed for production-level agent systems. Use the Python API for programmatic integration or the planned REST API (coming soon) for service deployment.

**How do I debug agent behavior?**
Use `print(agent)` and `print(agent.skills)` to inspect configuration. The `rich` library provides enhanced output formatting. Check predictions with `agent.run(test_df)`.

### Troubleshooting

**Why is my agent not learning?**
- Verify API key is set correctly
- Check that ground truth data format matches skill's templates
- Ensure model supports the task complexity (try GPT-4o for complex tasks)
- Increase `learning_iterations` if accuracy threshold isn't met

**What if predictions are incorrect?**
- Review skill `instructions` for clarity
- Add more ground truth examples covering edge cases
- Try a stronger model (GPT-4o vs GPT-3.5-turbo)
- Adjust `accuracy_threshold` based on acceptable quality

**Where can I get help?**
- Discord: https://discord.gg/QBtgTbXTgU
- GitHub Issues: https://github.com/HumanSignal/Adala/issues
- Examples: `./examples/` directory with quickstart notebooks

## 🗺 Roadmap

- [x] Low-level skill management (i.e. agent.get_skill("name")) [COMPLETE @niklub]
- [ ] Make every notebook example to run in Google Collab and add a badge into README
- [ ] Extend environment with one more example
- [ ] Multi-task learning (learn multiple skills at once)
- [ ] Calculate and store top line Agent metrics (predictions created, runtime executions, learning loops, etc)
- [ ] Create Named Entity Recognition Skill
- [ ] Command line utility (see the source for this readme for example)
- [ ] REST API to interact with Adala
- [ ] Vision and multi-modal agent skills

## 🤩 Contributing to Adala

Enhance skills, optimize runtimes, or pioneer new agent types. Whether you're
crafting nuanced tasks, refining computational environments, or sculpting specialized agents for unique domains, your
contributions will power Adala's evolution. Join us in shaping the future of intelligent systems and making Adala more
versatile and impactful for users across the globe.

[Read more](./CONTRIBUTION.md) here.

## 💬 Support

Do you need help or are you looking to engage with community? Check out [Discord channel](https://discord.gg/QBtgTbXTgU)!
Whether you have questions, need clarification, or simply want to discuss topics related to the project, the Discord community is welcoming!
