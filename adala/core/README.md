# Adala Core - Simplified Data Processing with LLMs

This module provides a streamlined interface for processing tabular data through LLMs, removing unnecessary abstraction layers while preserving the core functionality.

## Key Components

### DataTable

A direct wrapper around pandas DataFrame that provides a clean interface for working with tabular data:

```python
from adala.utils.internal_data import DataTable

# Create from list of dicts
data = DataTable([
    {"text": "Sample text 1", "label": "positive"},
    {"text": "Sample text 2", "label": "negative"}
])

# Convert pandas DataFrame to DataTable
import pandas as pd
df = pd.DataFrame({"text": ["Sample 1", "Sample 2"]})
data = DataTable.from_dataframe(df)

# Get records as list of dicts
records = data.to_records()
```

### BatchLLMRuntime

A runtime for processing batches of data through LLMs with efficient batching and concurrency:

```python
from adala.runtimes.batch_llm import BatchLLMRuntime
from pydantic import BaseModel, Field

class ClassificationOutput(BaseModel):
    label: str = Field(..., description="Classification label")

runtime = BatchLLMRuntime(
    model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.0,
    batch_size=10,
    concurrency=4
)

results = runtime.process_batch(
    data=my_data,
    prompt_template="Classify the following text as positive or negative: {text}",
    response_model=ClassificationOutput
)
```

### DataProcessor

A high-level interface for data processing tasks:

```python
from adala.core.processor import DataProcessor
from pydantic import BaseModel, Field

class NamedEntityOutput(BaseModel):
    person: str = Field(..., description="Name of person mentioned")
    location: str = Field(..., description="Location mentioned")

processor = DataProcessor(
    prompt_template="Extract the person and location from this text: {text}",
    response_model=NamedEntityOutput,
    model="gpt-4"
)

# Process a batch
results = processor.process(my_data)

# Process asynchronously
import asyncio
async def run():
    results = await processor.aprocess(my_data)
```

### Classifier

A specialized processor for classification tasks:

```python
from adala.core.processor import Classifier

classifier = Classifier(
    instructions="Classify the text as one of the given categories",
    labels=["Sports", "Politics", "Technology", "Entertainment"],
    input_field="content",
    output_field="category"
)

# Add context to be included in all prompts
classifier.add_context(
    examples=["Example 1: Sports", "Example 2: Politics"]
)

# Process data
results = classifier.process(my_data)
```

## Benefits

- **Direct Data Access**: Work directly with your tabular data
- **Minimal Configuration**: Less boilerplate, more focused on your task
- **Efficient Batch Processing**: Built-in batching and parallelism
- **Structured Output**: Pydantic models ensure properly formatted results
- **Async Support**: Process data asynchronously for greater throughput

## Example Workflow

```python
import pandas as pd
from adala.core.processor import Classifier

# Load your data
df = pd.DataFrame([
    "New iPhone announced with improved camera",
    "Government passes new healthcare legislation",
    "Lakers win championship after close game"
], columns=["text"])

# Create and configure classifier
classifier = Classifier(
    instructions="Classify the news headline into the appropriate category.",
    labels=["Technology", "Politics", "Sports", "Entertainment"],
    model="gpt-3.5-turbo"
)

# Process the data
results = classifier.process(df)
print(results)
```