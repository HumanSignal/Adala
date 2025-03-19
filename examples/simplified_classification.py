"""
Simplified Classification Example

This example demonstrates the simplified API for classification tasks,
which provides a more direct way to process tabular data through LLMs.
"""

import pandas as pd
from pydantic import BaseModel, Field
from adala import Classifier, DataTable  # Import directly from top-level package

# Create sample data
train_df = pd.DataFrame([
    ["The mic is great.", "Subjective"],
    ["Will order from them again!", "Subjective"],
    ["Not loud enough and doesn't turn on like it should.", "Objective"],
    ["The phone doesn't seem to accept anything except CBR mp3s.", "Objective"],
    ["All three broke within two months of use.", "Objective"]
], columns=["text", "ground_truth"])

# Create test data
test_df = pd.DataFrame([
    "Doesn't hold charge.",
    "Excellent bluetooth headset",
    "I love this thing!",
    "VERY DISAPPOINTED."
], columns=['text'])

# Create a classifier with minimal configuration
classifier = Classifier(
    instructions="Classify a product review as either expressing 'Subjective' or 'Objective' statements.",
    labels=["Subjective", "Objective"],
    input_field="text",
    output_field="prediction",
    description="Subjectivity classification for product reviews",
    model="gpt-3.5-turbo",
    temperature=0.0,
    verbose=True
)

# Add context about what subjective vs objective means
classifier.add_context(
    definition="""
    Subjective statements express personal feelings, emotions, or preferences.
    Objective statements describe factual details about product functionality or performance,
    even when based on personal experiences.
    """
)

# Process the test data
results = classifier.process(test_df)

# Print the results
print("\nClassification Results:")
print(results[["text", "prediction"]])

"""
Example of using a custom response model:
"""

# Define a custom response model
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(..., description="The sentiment of the text (Positive, Negative, or Neutral)")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Reasoning behind the sentiment classification")

# Create a data processor for sentiment analysis
from adala import DataProcessor

sentiment_analyzer = DataProcessor(
    prompt_template="""
    Analyze the sentiment of the following text and determine if it's Positive, Negative, or Neutral.
    
    Text: {text}
    
    Provide your sentiment analysis with a confidence score and reasoning.
    """,
    response_model=SentimentAnalysis,
    model="gpt-3.5-turbo",
    temperature=0.2
)

# Process the sentiment
sentiment_results = sentiment_analyzer.process(test_df[:2])

# Print the sentiment results
print("\nSentiment Analysis Results:")
for i, row in sentiment_results.iterrows():
    print(f"\nText: {row['text']}")
    print(f"Sentiment: {row['sentiment']} (Confidence: {row['confidence']:.2f})")
    print(f"Reasoning: {row['reasoning']}")

"""
Example of asynchronous processing:
"""

import asyncio

async def run_async_example():
    print("\nRunning Async Example...")
    async_results = await classifier.aprocess(test_df[2:])
    print("\nAsync Classification Results:")
    print(async_results[["text", "prediction"]])

if __name__ == "__main__":
    # Run the async example
    asyncio.run(run_async_example())