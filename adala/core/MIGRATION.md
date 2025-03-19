# Migration Guide: Transitioning to the Simplified API

This guide will help you migrate from the original Adala API to the simplified core API. The new API offers a more direct way to process tabular data through LLMs in batch mode, with less boilerplate and abstraction.

## Mapping Between APIs

| Original API | Simplified API | Notes |
|--------------|----------------|-------|
| `Agent` | `DataProcessor` | The main entry point for processing data |
| `ClassificationSkill` | `Classifier` | Specialized processor for classification tasks |
| `LabelStudioSkill` | `LabelStudioProcessor` | Processor for Label Studio annotations |
| `StaticEnvironment` | Direct pandas/DataTable | Work directly with your data |
| `OpenAIChatRuntime` | `BatchLLMRuntime` | Streamlined runtime for batch processing |
| `InternalDataFrame` | `DataTable` | Enhanced pandas DataFrame for batch operations |

## Code Examples

### Classification Example

#### Original API:

```python
from adala.agents import Agent
from adala.environments import StaticEnvironment
from adala.skills import ClassificationSkill
from adala.runtimes import OpenAIChatRuntime
import pandas as pd

df = pd.DataFrame([
    ["The product works great", "Positive"],
    ["Terrible quality, avoid", "Negative"]
], columns=["text", "sentiment"])

agent = Agent(
    skills=ClassificationSkill(
        name='sentiment',
        instructions="Classify text as positive, negative or neutral.",
        labels={'sentiment': ["Positive", "Negative", "Neutral"]},
        input_template="Text: {text}",
        output_template="Sentiment: {sentiment}"
    ),
    environment=StaticEnvironment(
        df=df,
        ground_truth_columns={'sentiment': 'sentiment'}
    ),
    runtimes = {
        'openai': OpenAIChatRuntime(model='gpt-3.5-turbo'),
    },
    default_runtime='openai'
)

# Train the agent
agent.learn(learning_iterations=3)

# Run prediction on new data
test_df = pd.DataFrame(["This is a new product"], columns=["text"])
predictions = agent.run(test_df)
```

#### Simplified API:

```python
from adala.core import Classifier
import pandas as pd

# Training data
df = pd.DataFrame([
    ["The product works great", "Positive"],
    ["Terrible quality, avoid", "Negative"]
], columns=["text", "sentiment"])

# Create classifier
classifier = Classifier(
    instructions="Classify text as positive, negative or neutral.",
    labels=["Positive", "Negative", "Neutral"],
    model="gpt-3.5-turbo"
)

# No separate training step required; examples can be added as context
classifier.add_context(
    examples=[
        {"text": "The product works great", "label": "Positive"},
        {"text": "Terrible quality, avoid", "label": "Negative"}
    ]
)

# Run prediction on new data
test_df = pd.DataFrame(["This is a new product"], columns=["text"])
predictions = classifier.process(test_df)
```

### Custom Processing Example

#### Original API:

```python
from adala.agents import Agent
from adala.skills import TransformSkill
from adala.runtimes import OpenAIChatRuntime
from pydantic import BaseModel, Field

class EntityExtraction(BaseModel):
    person: str = Field(..., description="Person mentioned in text")
    location: str = Field(..., description="Location mentioned in text")

agent = Agent(
    skills=TransformSkill(
        name='entity_extraction',
        instructions="Extract person and location entities from text.",
        input_template="Text: {text}",
        output_template="Entities: {entities}",
        response_model=EntityExtraction
    ),
    runtimes = {
        'openai': OpenAIChatRuntime(model='gpt-4'),
    },
    default_runtime='openai'
)

# Run on data
import pandas as pd
df = pd.DataFrame(["John visited Paris last summer"], columns=["text"])
results = agent.run(df)
```

#### Simplified API:

```python
from adala.core import DataProcessor
from pydantic import BaseModel, Field
import pandas as pd

class EntityExtraction(BaseModel):
    person: str = Field(..., description="Person mentioned in text")
    location: str = Field(..., description="Location mentioned in text")

processor = DataProcessor(
    prompt_template="Extract person and location entities from this text: {text}",
    response_model=EntityExtraction,
    model="gpt-4"
)

# Run on data
df = pd.DataFrame(["John visited Paris last summer"], columns=["text"])
results = processor.process(df)
```

### Label Studio Example

#### Original API:

```python
from adala.agents import Agent
from adala.skills import LabelStudioSkill
from adala.runtimes import OpenAIChatRuntime
import pandas as pd

# Define the Label Studio configuration
label_config = """
<View>
  <Text name="text" value="$text"/>
  <Labels name="ner_tags" toName="text">
    <Label value="Person"/>
    <Label value="Organization"/>
    <Label value="Location"/>
  </Labels>
</View>
"""

# Create the agent with Label Studio skill
agent = Agent(
    skills=LabelStudioSkill(
        name='ner_tagger',
        instructions="Annotate the text with named entities.",
        label_config=label_config
    ),
    runtimes = {
        'openai': OpenAIChatRuntime(model='gpt-3.5-turbo'),
    },
    default_runtime='openai'
)

# Run on data
df = pd.DataFrame(["John works at Apple in San Francisco"], columns=["text"])
results = agent.run(df)
```

#### Simplified API:

```python
from adala import LabelStudioProcessor
import pandas as pd

# Define the Label Studio configuration
label_config = """
<View>
  <Text name="text" value="$text"/>
  <Labels name="ner_tags" toName="text">
    <Label value="Person"/>
    <Label value="Organization"/>
    <Label value="Location"/>
  </Labels>
</View>
"""

# Create the processor
processor = LabelStudioProcessor(
    label_config=label_config,
    instructions="Annotate the text with named entities.",
    model="gpt-3.5-turbo"
)

# Run on data
df = pd.DataFrame(["John works at Apple in San Francisco"], columns=["text"])
results = processor.process(df)
```

## Async Processing

The simplified API supports asynchronous processing out of the box:

```python
import asyncio
from adala.core import Classifier
import pandas as pd

classifier = Classifier(
    instructions="Classify the sentiment of the text.",
    labels=["Positive", "Negative", "Neutral"],
    model="gpt-3.5-turbo"
)

async def process_data():
    df = pd.DataFrame(["I love this product", "This is terrible"], columns=["text"])
    results = await classifier.aprocess(df)
    return results

# Run async function
results = asyncio.run(process_data())
```

## Benefits of Migration

1. **Less Boilerplate**: Write less code to accomplish the same tasks
2. **Better Performance**: Direct batch processing without unnecessary wrapper code
3. **Simpler Mental Model**: Work directly with your data instead of through multiple abstraction layers
4. **Async Support**: First-class support for asynchronous processing
5. **Maintainability**: Less code means fewer bugs and easier maintenance

## Common Migration Patterns

1. **Replace Agent with DataProcessor or Classifier**: Choose the appropriate processor for your task
2. **Eliminate Environment**: Work directly with your data in pandas or DataTable format
3. **Convert Skills to Prompt Templates**: Move your skill logic into prompt templates and response models
4. **Replace Runtime Configuration**: Use the simplified BatchLLMRuntime with concurrency settings
5. **Use add_context() for Examples**: Instead of StaticEnvironment, add examples via context