import pytest
from adala.utils.parse import partial_str_format


def test_partial_string_format():
    # Test basic string formatting with all variables present
    result = partial_str_format(
        "My name is {input_name} and I am {input_age} years old.",
        input_name="Carla",
        input_age=25,
    )
    assert result == "My name is Carla and I am 25 years old."

    # Test with missing variables - should keep original placeholder
    result = partial_str_format(
        "My name is {input_name} and I am {input_age} years old with {brackets} and {brackets2}.",
        input_name="Carla",
        input_age=25,
    )
    assert (
        result
        == "My name is Carla and I am 25 years old with {brackets} and {brackets2}."
    )

    # Test with format specifiers
    result = partial_str_format(
        "My name is {input_name} and I am {input_age:02d} years old with {brackets:.2f} and {brackets2:invalid_format_spec} and {input_name:invalid_format_spec}.",
        input_name="Carla",
        input_age=25,
    )
    assert (
        result
        == "My name is Carla and I am 25 years old with {brackets:.2f} and {brackets2:invalid_format_spec} and {input_name:invalid_format_spec}."
    )

    # Test with empty string
    result = partial_str_format("")
    assert result == ""

    # Test with no placeholders
    result = partial_str_format("Hello world")
    assert result == "Hello world"

    # Test with only missing placeholders
    result = partial_str_format("{missing1} and {missing2}")
    assert result == "{missing1} and {missing2}"

    # Test with unmatched brackets
    result = partial_str_format("{ {text}", text="test")
    assert result == "{ test"

    # Test adversarial example
    result = partial_str_format(
        '{"key": "value", "text": "{text}", "unused1": "{unused}", "nested": {"subkey": "{more_unexpected}"}, "break": "\\" \' \\n \\t \\b \\f \\r \\\\ \\/ {unmatched", "unicode": "\uD83D\uDE00", "null": null, "number": 123, "array": [1, 2, "{array_item}"], "weird": "\u0000\u001F"}',
        text="test",
    )
    assert (
        result
        == '{"key": "value", "text": "test", "unused1": "{unused}", "nested": {"subkey": "{more_unexpected}"}, "break": "\\" \' \\n \\t \\b \\f \\r \\\\ \\/ {unmatched", "unicode": "\uD83D\uDE00", "null": null, "number": 123, "array": [1, 2, "{array_item}"], "weird": "\u0000\u001F"}'
    )

    # Test larger prompt
    prompt = """
Given the following product review:

**Title:** {Title}

**Review Text:** {Text}

Your task is to perform the following actions:
1. **Highlight Entities:** Identify and label entities in the review text. The entities can be of the following types: Animal, Emotion, Product, and Location. Provide the start and end indices of each entity in the text along with the corresponding label.
2. **Choose Review Category:** Based on the content of the review, select one or more categories from the following options: baby products, beauty, grocery gourmet food, health personal care, pet supplies, toys games.
3. **Classify Review Sentiment:** Determine the sentiment of the review as either positive or negative.
4. **Provide Reasoning:** Explain your reasoning for the sentiment classification in a concise manner.

Please format your output in JSON with the following keys:
- **label:** An array of objects, each containing the start and end indices of the identified entities, their labels, and the corresponding text.
- **category:** An array of selected categories for the review.
- **satisfaction:** The sentiment classification of the review (positive or negative).
- **reasoning:** A brief explanation of the reasoning behind the sentiment classification.

### Examples:
1. Input: {Title: "PetSafe StayWell Pet door", Text: "We have only installed about 2 weeks but ..."}
   Output: {"label": [{"start": 8, "end": 12, "labels": ["Emotion"], "text": "only"}], "category": ["pet supplies"], "satisfaction": "negative", "reasoning": "The review expresses dissatisfaction with the product after a short usage period."}

2. Input: {Title: "Best Dog Food Ever", Text: "My dog loves this food! It's amazing!"}
   Output: {"label": [{"start": 0, "end": 3, "labels": ["Product"], "text": "Dog Food"}], "category": ["pet supplies"], "satisfaction": "positive", "reasoning": "The review expresses a positive sentiment towards the product, indicating that the dog enjoys it."}
"""
    data = {
        "Text": "We have only installed about 2 weeks but ...",
        "Title": "PetSafe StayWell Pet door",
    }
    result = partial_str_format(prompt, **data)
    assert (
        result
        == """
Given the following product review:

**Title:** PetSafe StayWell Pet door

**Review Text:** We have only installed about 2 weeks but ...

Your task is to perform the following actions:
1. **Highlight Entities:** Identify and label entities in the review text. The entities can be of the following types: Animal, Emotion, Product, and Location. Provide the start and end indices of each entity in the text along with the corresponding label.
2. **Choose Review Category:** Based on the content of the review, select one or more categories from the following options: baby products, beauty, grocery gourmet food, health personal care, pet supplies, toys games.
3. **Classify Review Sentiment:** Determine the sentiment of the review as either positive or negative.
4. **Provide Reasoning:** Explain your reasoning for the sentiment classification in a concise manner.

Please format your output in JSON with the following keys:
- **label:** An array of objects, each containing the start and end indices of the identified entities, their labels, and the corresponding text.
- **category:** An array of selected categories for the review.
- **satisfaction:** The sentiment classification of the review (positive or negative).
- **reasoning:** A brief explanation of the reasoning behind the sentiment classification.

### Examples:
1. Input: {Title: "PetSafe StayWell Pet door", Text: "We have only installed about 2 weeks but ..."}
   Output: {"label": [{"start": 8, "end": 12, "labels": ["Emotion"], "text": "only"}], "category": ["pet supplies"], "satisfaction": "negative", "reasoning": "The review expresses dissatisfaction with the product after a short usage period."}

2. Input: {Title: "Best Dog Food Ever", Text: "My dog loves this food! It's amazing!"}
   Output: {"label": [{"start": 0, "end": 3, "labels": ["Product"], "text": "Dog Food"}], "category": ["pet supplies"], "satisfaction": "positive", "reasoning": "The review expresses a positive sentiment towards the product, indicating that the dog enjoys it."}
"""
    )
