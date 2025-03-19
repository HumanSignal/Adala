# Adala - Simplified Data Processing with LLMs

This is a refactored version of the Adala framework that provides a streamlined interface for processing tabular data through LLMs.

## Key Components

- **DataTable**: A lightweight wrapper around pandas DataFrame
- **BatchLLMRuntime**: Efficient batch processing of data through LLMs
- **DataProcessor**: High-level interface for data processing tasks
- **Classifier**: Specialized processor for classification tasks

## Getting Started

```python
import pandas as pd
from adala import Classifier

# Create sample data
df = pd.DataFrame([
    "Not loud enough and doesn't turn on like it should.",
    "The product works perfectly fine.",
    "I absolutely love this device!"
], columns=["text"])

# Create a classifier
classifier = Classifier(
    instructions="Classify product reviews as positive, negative, or neutral.",
    labels=["Positive", "Negative", "Neutral"],
    model="gpt-3.5-turbo"
)

# Process the data
results = classifier.process(df)
print(results[["text", "label"]])
```

## Legacy Components

The original Adala components (Agent, Skills, Environments, Memories) are still available but deprecated. They have been moved to the `adala.legacy` module and will be removed in a future version.

For migration guidance, see [MIGRATION.md](./core/MIGRATION.md)