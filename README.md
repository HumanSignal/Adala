# ADALA
ADALA: Autonomous Data Labeling Agent

## Quick Start

### Install Label Studio

```bash
pip install label-studio
```

and start it with `label-studio`

Now create a new project `Create > New Project > Labeling Setup > Natural Language Processing > Text Classification`
Get the project ID `label_studio_project_id` from the URL, it will be used later.

### Install ADALA

```bash
git clone https://github.com/HumanSignal/ADALA.git
cd ADALA/
pip install -e .
```

### Run ADALA

```python
import os
import adala
import pandas as pd

os.environ['LABEL_STUDIO_API_TOKEN'] = 'your_token'
os.environ['LABEL_STUDIO_HOST'] = 'http://localhost:8080'
os.environ['OPENAI_API_KEY'] = 'your_key'


df = pd.read_csv('data.csv')

results = adala.label(df, label_studio_project_id, initial_instructions='Go go!')
results['predicted_df']
```