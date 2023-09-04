# ADALA
ADALA: Automated Data Labeling Framework.

ADALA is a framework for automated data labeling. It uses a combination of Large Language Models (LLMs) and Active Learning (AL) to label data. It is designed to be used with [Label Studio](https://labelstud.io/) to provide a human-in-the-loop data labeling experience.

Here is what ADALA does:
- [LLM instructions generation](#llm-instructions-generation)
- [Predicting dataset with LLM given the instructions](#applying-llm-to-the-dataset-given-the-instructions)
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


## LLM instructions generation

ADALA uses Large Language Models (LLMs) to generate instructions for data labeling. You need to have an [OpenAI API](https://platform.openai.com/) key to use ADALA.

```bash
export OPENAI_API_KEY=your_key
```

Load the data into a pandas DataFrame:
```python
import pandas as pd
df = pd.read_csv('dataset.csv')
```

The following method allows you to finetune instructions to classify each row in the DataFrame, given the ground truth labels in the specified column:
```python
import adala as ad

instructions = ad.generate_instructions(
    df,
    ground_truth_column='label'
)
```

## Applying LLM to the dataset given the instructions

ADALA used optimized batch inference to run LLM on the dataset. 

Create LLM predictor:
```python
predictor = ad.LLMPredictor(model='gpt3')
```

Predict the dataset:
```python
predicted_df = predictor.predict(
    df,
    instructions=instructions,
    prediction_column='predictions'
)
predicted_df['predictions']
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
    df,
    label_studio_project_id=project_id,
    output_column='autolabel'
)
labeled_df['autolabel']
```

## LLM uncertainty estimation

ADALA can be used to estimate LLM uncertainty for each row in the dataset. It is useful if you want to detect hallucinations or other forms of LLM errors.

```python
uncertainty_df = ad.estimate_uncertainty(
    df,
    instructions=instructions,
    prediction_column='predictions',
    uncertainty_column='uncertainty'
)
uncertainty_df['uncertainty']
```
