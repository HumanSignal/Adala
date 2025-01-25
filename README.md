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

# Meet Adala 🛠️📊

Adala gives you well-curated, actionable agentic workflows to take full control of your ML data processing. It’s designed to simplify complex tasks and help you focus on results instead of the details.

## What can Adala help you with?

* 🏷️ Labeling and prepping data for training AI models
* 📂 Organizing and managing data across your projects
* 🔍 Exploring and analyzing datasets to uncover insights
* 📊 Monitoring AI predictions for accuracy and fairness
* 🛠️ Creating new datasets when you need them
* 🤖 Testing and improving AI models (like LLMs)

It seamlessly integrates data providers, data labeling, visualization tools, and ML pipelines, so you can focus on your data and results. 🎉

# User Guide 📖

## Installation

```
pip install adala
```

### Install Adala with Claude

```
adala install --with-claude
```

Install and run [Claude desktop](https://claude.ai/download).

### Install Adala standalone

Launch Adala server:
```
adala run
```

Launch Adala client:
```
adala client
```

## Usage
Use any [workflow from the list](#workflows) and paste it into the agentic chat (e.g. Claude) to execute it.


# Workflows 🧰

Workflow is just a simple text instruction that describes what you want to do with your data.  Feel free to use and modify any of the provided workflows or create your own:

#### Create and autolabel LLM fairness dataset
```
Create a new labeling project to evaluate large language models focusing on LLM fairness. Include a few demo examples in this project to address fairness issues. Automatically label the provided data. Share the link to this project for human review.
```

#### Create an annotation project from data on S3 bucket
```
The goal of the project is {project_goal}. Create a new annotation project from JSON files on my S3 bucket: {bucket_name}. Prioritize annotation quality over efficiency. Share the link to the project for human review.
```

#### Monitor model predictions
```
Track predictions from {model_name} over the last {time_period}. Evaluate for consistency and accuracy. Highlight areas where prediction confidence is low or mismatched with expected results. Generate a report and notify the team.  
```

#### Generate a synthetic dataset
```
Create a synthetic dataset for {use_case}. Include diverse examples covering edge cases. Ensure the data aligns with real-world scenarios. Validate the generated dataset against existing benchmarks.
```

#### Explore dataset for missing values and patterns
```
Analyze {dataset_name} for missing values, anomalies, and patterns. Generate a summary report with visualizations. Highlight any inconsistencies or potential data quality issues.  
```

#### Evaluate LLM on question answering
```
Create a new project to evaluate {LLM_name} on question answering. Use the provided dataset and score each response on relevance and accuracy. Share results as a detailed performance report.  
```

#### Monitor annotation progress and labelers performance
```
Track the progress of the ongoing annotation project: {project_name}. Generate a daily summary of completed tasks, pending work, and quality metrics. Notify team members of any delays or issues.  
```

#### Visualize customer reports trends
```
Visualize the trends of customer reports over the last {time_period}. Generate a report with visualizations and insights. Highlight any significant changes or patterns.
```

Access [the full list of workflows](database-link) and submit your own! 📝

## 🗺 Roadmap

- [ ] More Workflows
- [ ] More Actions
- [ ] More Clients
- [ ] More Data Providers and integrations

## 🤩 Contributing to Adala

Workflows are based on Agentic tools, or `@actions`. To define a new action, decorate you typed method:

```python
from adala import action

@action()
def finetune_my_model(dataset_path: str, epochs: int = 10):
    ...
    return "Done!"
```

You can use actions to:
- integrate with your data or model provider
- perform arbitrary computation and data processing
- guide Adala agent to perform complex tasks

Adala framework is compatible with [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)

[Read more](./CONTRIBUTION.md) here.

## 💬 Support

Do you need help or are you looking to engage with community? Check out [Discord channel](https://discord.gg/QBtgTbXTgU)!
Whether you have questions, need clarification, or simply want to discuss topics related to the project, the Discord community is welcoming!
