"""
Simplified Label Studio Example

This example demonstrates how to use the simplified LabelStudioProcessor
for annotation tasks using Label Studio's schema.
"""

import pandas as pd
from adala import LabelStudioProcessor, DataTable

# Define a simple Label Studio configuration for NER
LABEL_CONFIG = """
<View>
  <Text name="text" value="$text"/>
  <Labels name="ner_tags" toName="text">
    <Label value="Person" background="#FE9573"/>
    <Label value="Organization" background="#9013FE"/>
    <Label value="Location" background="#33D391"/>
  </Labels>
</View>
"""

# Create sample data
data = pd.DataFrame([
    {"text": "John works at Apple in San Francisco."},
    {"text": "Microsoft has an office in Seattle where Sarah is working."},
    {"text": "The president visited New York last week."}
])

# Create the Label Studio processor
processor = LabelStudioProcessor(
    label_config=LABEL_CONFIG,
    instructions="""
    Annotate the text with named entities.
    
    - Person: Names of people (e.g., "John", "Sarah")
    - Organization: Names of companies and organizations (e.g., "Apple", "Microsoft")
    - Location: Names of cities, countries, and other locations (e.g., "San Francisco", "Seattle")
    
    Please annotate all instances in the text.
    """,
    model="gpt-3.5-turbo",
    temperature=0.0,
    verbose=True
)

# Process the data
print("\nProcessing data with Label Studio schema...")
results = processor.process(data)

# Print the results in a readable format
print("\nAnnotation Results:")
for i, row in results.iterrows():
    print(f"\nText: {row['text']}")
    print("Entities:")
    for entity in row.get('ner_tags', []):
        if isinstance(entity, dict):
            print(f"  - {entity.get('text', 'N/A')} [{entity.get('labels', [''])[0]}] "
                  f"({entity.get('start', 'N/A')}-{entity.get('end', 'N/A')})")

"""
Example with image input:
"""

# Define a Label Studio configuration for image classification
IMAGE_LABEL_CONFIG = """
<View>
  <Image name="image" value="$image_url"/>
  <Choices name="image_category" toName="image">
    <Choice value="Animal"/>
    <Choice value="Landscape"/>
    <Choice value="Person"/>
    <Choice value="Food"/>
    <Choice value="Other"/>
  </Choices>
</View>
"""

# Create sample image data
image_data = pd.DataFrame([
    {"image_url": "https://example.com/images/cat.jpg"},
    {"image_url": "https://example.com/images/mountain.jpg"},
    {"image_url": "https://example.com/images/person.jpg"}
])

print("\n\nImage Classification Example:")
print("This example would work with actual image URLs and a vision model")
print(f"Label Studio Config: {IMAGE_LABEL_CONFIG}")
print(f"Sample Data: {image_data.head()}")
print("To run this example, you would need:\n1. Real image URLs\n2. A vision-capable model like gpt-4-vision")