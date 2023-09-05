# ADALA <img src="https://htx-pub.s3.amazonaws.com/samples/Adala.png" width="100" align="right"/>
Automated Data Labeling Framework. 

[![PyPI version](https://badge.fury.io/py/adala.svg)](https://badge.fury.io/py/adala)
[![Python version](https://img.shields.io/pypi/pyversions/adala.svg)](https://pypi.python.org/pypi/adala)
[![License](https://img.shields.io/pypi/l/adala.svg)](https://pypi.python.org/pypi/adala)


ADALA is a framework for automated data labeling. It uses a combination of Large Language Models (LLMs) autonomous agents and Active Learning (AL) to label data. It is designed to be used with [Label Studio](https://labelstud.io/) to provide a human-in-the-loop data labeling experience.

Here is what ADALA does:
- [LLM instructions generation](#llm-instructions-generation)
- [Predicting dataset with LLM](#predicting-dataset-with-llm)
- [Active learning with Human-in-the-Loop](#active-learning-with-human-in-the-loop)
- [LLM uncertainty estimation](#llm-uncertainty-estimation)


## Installation

Install ADALA:
```bash
git clone https://github.com/HumanSignal/ADALA.git
cd ADALA/
pip install -e .
```

If you're planning to use human-in-the-loop labeling, install Label Studio:
```bash
pip install label-studio
```

## Load dataset
ADALA works with datasets in various formats:
- [Pandas DataFrame](#pandas-dataframe)
- [Spark DataFrame](#spark-dataframe)

### Pandas DataFrame

Load the data into a pandas DataFrame:
```python
import pandas as pd
input_df = pd.read_csv('dataset.csv')
```

### Spark DataFrame

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
```


## Predicting dataset with LLM

ADALA inference is optimized to run in the batch mode - it is much faster to predict the whole dataset at once, rather than row-by-row.

Create LLM predictor:
```python
predictor = ad.LLMPredictor()
```

There are multiple LLM models available in the table below:
| Model    | Initialize predictor |
| -------- | ------- |
| [Any LangChain's LLM](https://python.langchain.com/docs/get_started/introduction.html) | `ad.LangChainLLMPredictor()`    |
| [HuggingFace TGI](https://huggingface.co/text-generation-inference) | `ad.HuggingFaceLLMPredictor()`     |
| [vLLM](https://vllm.ai/)    | `ad.VLLMPredictor()`    |
| [llama.cpp](https://github.com/ggerganov/llama.cpp)   | `ad.LlamaCppPredictor()`   |


Predict the dataset:
```python
predicted_df = predictor.predict(
    df=input_df,
    instruction='Predict sentiment',
    labels=['positive', 'negative'],
    prediction_column='predictions'
)
predicted_df['predictions']
```


## LLM instructions generation

ADALA can generate optimal LLM instructions for data labeling. You need to have an [OpenAI API](https://platform.openai.com/) key to use ADALA.

```bash
export OPENAI_API_KEY=your_key
```

The following method allows you to finetune instructions to classify each row in the DataFrame, given the ground truth labels in the specified column:
```python
import adala as ad

instructions = ad.generate_instructions(
    df=input_df,
    ground_truth_column='label'
)
```

## Active learning with Human-in-the-Loop

Combining instructions generation and dataset prediction, ADALA can be used to create a human-in-the-loop automated data labeling experience with Label Studio.

First [create a Label Studio project](https://labelstud.io/guide/setup_project).

> Note: Currently ADALA is designed to work with Text Classification projects. Go to `Labeling Setup > Natural Language Processing > Text Classification`. Change label names to match your dataset labels.

Get the project ID `project_id` from the URL, it will be used later.

Setup environment variables with [Label Studio API token](https://labelstud.io/guide/api#Authenticate-to-the-API) and Label Studio host:
```bash
export LABEL_STUDIO_API_TOKEN=your_token
export LABEL_STUDIO_HOST=http://localhost:8080
```

Run ADALA human-in-the-loop labeling:
```python
labeled_df = ad.human_in_the_loop(
    df=input_df,
    label_studio_project_id=project_id,
    output_column='predictions'
)
labeled_df['predictions']
```

## LLM uncertainty estimation

ADALA can be used to estimate LLM uncertainty for each row in the dataset. It is useful if you want to detect hallucinations or other forms of LLM errors.

```python
uncertainty_df = ad.estimate_uncertainty(
    df=labeled_df,
    instructions=instructions,
    prediction_column='predictions',
    output_column='uncertainty'
)
uncertainty_df['uncertainty']
```
