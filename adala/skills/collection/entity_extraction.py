import logging
import pandas as pd
import jsonschema
from jsonschema import validate
from adala.runtimes import Runtime, AsyncRuntime
from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field, model_validator
from typing import List, Type, Optional, Dict, Any

from adala.utils.internal_data import InternalDataFrame
from adala.utils.pydantic_generator import field_schema_to_pydantic_class

logger = logging.getLogger(__name__)


def validate_output_format_for_ner_tag(
    df: InternalDataFrame, input_field_name: str, output_field_name: str
):
    """
    The output format for Labels is:
    {
        "start": start_idx,
        "end": end_idx,
        "text": text,
        "labels": [label1, label2, ...]
    }
    Sometimes the model cannot populate "text" correctly, but this can be fixed deterministically.
    """
    for i, row in df.iterrows():
        if row.get("_adala_error"):
            logger.warning(f"Error in row {i}: {row['_adala_message']}")
            continue
        text = row[input_field_name]
        entities = row[output_field_name]
        if entities is not None:
            for entity in entities:
                corrected_text = text[entity["start"] : entity["end"]]
                if entity.get("text") is None:
                    entity["text"] = corrected_text
                elif entity["text"] != corrected_text:
                    # this seems to happen rarely if at all in testing, but could lead to invalid predictions
                    logger.error(
                        "text and indices disagree for a predicted entity: Entity: %s, Corrected text: %s",
                        entity["text"],
                        corrected_text,
                    )
    return df


def find_all_occurrences(text: str, substring: str) -> List[int]:
    """Find all starting indices of substring in text (case-insensitive)."""
    occurrences = []
    text_lower = text.lower()
    substring_lower = substring.lower()
    start = 0
    while True:
        idx = text_lower.find(substring_lower, start)
        if idx == -1:
            break
        occurrences.append(idx)
        start = idx + 1
    return occurrences


def is_word_boundary(text: str, start: int, end: int) -> bool:
    """
    Check if the substring at text[start:end] has word boundaries on both sides.
    A word boundary is the start/end of string, whitespace, or punctuation.
    """
    # Check left boundary
    if start > 0:
        char_before = text[start - 1]
        if char_before.isalnum():
            return False

    # Check right boundary
    if end < len(text):
        char_after = text[end]
        if char_after.isalnum():
            return False

    return True


def find_best_occurrence(
    text: str,
    substring: str,
    hint_start: Optional[int],
    used_ranges: List[tuple],
) -> Optional[int]:
    """
    Find the best occurrence of substring in text, using multiple strategies:
    1. Prefer word-boundary matches (standalone words) over partial matches within words
    2. If hint_start is provided, prefer occurrences closest to the hint
    3. Avoid already used ranges

    Args:
        text: The input text to search in
        substring: The substring to find
        hint_start: The model's predicted start index (used as a hint for finding closest match)
        used_ranges: List of (start, end) tuples representing already assigned entity ranges

    Returns:
        The start index of the best occurrence, or None if not found
    """
    occurrences = find_all_occurrences(text, substring)
    if not occurrences:
        return None

    # Filter out occurrences that overlap with already used ranges
    substring_len = len(substring)
    valid_occurrences = []
    for occ_start in occurrences:
        occ_end = occ_start + substring_len
        # Check if this occurrence overlaps with any used range
        overlaps = False
        for used_start, used_end in used_ranges:
            # Check for overlap: ranges overlap if one starts before the other ends
            if occ_start < used_end and occ_end > used_start:
                overlaps = True
                break
        if not overlaps:
            valid_occurrences.append(occ_start)

    if not valid_occurrences:
        return None

    # Separate into word-boundary matches and partial matches
    word_boundary_matches = []
    partial_matches = []
    for occ_start in valid_occurrences:
        occ_end = occ_start + substring_len
        if is_word_boundary(text, occ_start, occ_end):
            word_boundary_matches.append(occ_start)
        else:
            partial_matches.append(occ_start)

    # Prefer word-boundary matches; fall back to partial matches if none exist
    candidates = word_boundary_matches if word_boundary_matches else partial_matches

    # If we have a hint from the model, find the closest occurrence to it
    if hint_start is not None:
        return min(candidates, key=lambda x: abs(x - hint_start))

    # Otherwise, return the first valid occurrence
    return candidates[0]


def extract_indices(
    df,
    input_field_name,
    output_field_name,
    quote_string_field_name="quote_string",
    labels_field_name="label",
):
    """
    Give the input dataframe with "text" column and "entities" column of the format
    ```
    [{"quote_string": "entity_1"}, {"quote_string": "entity_2"}, ...]
    ```
        extract the indices of the entities in the input text and put indices in the "entities" column:
        ```
        [{"quote_string": "entity_1", "start": 0, "end": 5}, {"quote_string": "entity_2", "start": 10, "end": 15}, ...]
        ```

    If the model provides start/end indices, they are used as hints to find the closest
    matching occurrence in the text (since model indices are often incorrect but close).
    """
    for i, row in df.iterrows():
        if row.get("_adala_error"):
            logger.warning(f"Error in row {i}: {row['_adala_message']}")
            continue
        text = row[input_field_name]
        entities = row[output_field_name] or []
        to_remove = []
        used_ranges = []  # Track (start, end) of already assigned entities

        for entity in entities:
            ent_str = entity[quote_string_field_name]

            # Use model's predicted start index as a hint (if available)
            # The model's indices are often wrong but typically close to the correct position
            hint_start = entity.get("start")

            start_idx = find_best_occurrence(text, ent_str, hint_start, used_ranges)

            if start_idx is None:
                # Entity string not found in text (or all occurrences already used)
                to_remove.append(entity)
            else:
                end_idx = start_idx + len(ent_str)
                entity["start"] = start_idx
                entity["end"] = end_idx
                used_ranges.append((start_idx, end_idx))

        for entity in to_remove:
            entities.remove(entity)
    return df


def validate_schema(schema: Dict[str, Any]):
    expected_schema = {
        "type": "object",
        "patternProperties": {
            # name of the output field
            ".*": {
                "type": "object",
                "properties": {
                    # "type": "array"
                    "type": {"type": "string", "enum": ["array"]},
                    "description": {"type": "string"},
                    # "items": {"type": "object"} - one or two properties
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["object"]},
                            "properties": {
                                "type": "object",
                                "patternProperties": {
                                    ".*": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["string"],
                                            },
                                            "description": {"type": "string"},
                                            "enum": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                        "required": ["type"],
                                    }
                                },
                                "minProperties": 1,
                                "maxProperties": 2,
                            },
                        },
                        "required": ["type", "properties"],
                    },
                },
                "required": ["type", "items"],
            }
        },
        "additionalProperties": False,
    }

    try:
        validate(instance=schema, schema=expected_schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ValueError(f"Invalid schema: {e.message}")


class EntityExtraction(TransformSkill):
    """
    Extract entities from the input text.
    Example of the input and output:
    **Input**:
    ```
    {"text": "The quick brown fox jumps over the lazy dog."}
    ```
    **Output field schema:**
    ```
    {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote_string": {
                        "type": "string",
                        "description": "The text of the entity extracted from the input document."
                    },
                    "label": {
                        "type": "string",
                        "description": "The label assigned to the entity.",
                        "enum": ["COLOR", "ANIMAL"]
                    }
                }
            }
        }
    }
    ```
    **Output**:
    ```
    {"entities": [
        {"quote_string": "brown", "label": "COLOR", "start": 10, "end": 15},
        {"quote_string": "fox", "label": "ANIMAL", "start": 16, "end": 19},
    ]}
    ```

    Attributes:
    - `name` (str): The name of the skill.
    - `input_template` (str): The template of the input.
    - `field_schema` (Optional[Dict[str, Any]]): The schema of the output field.
    - `labels` (Optional[List[str]]): (deprecated, use `field_schema` instead)
                                    The list of labels assigned to the entities. For example, ["COLOR", "ANIMAL"].
                                    If not provided, no labels will be assigned.
    - `output_template` (str): (deprecated, use `field_schema` instead)
                                The template of the output. For example, "Extracted entities: {entities}".
    - `response_model` (Optional[Type[BaseModel]]): The Pydantic model of the response.
                                                    If not provided, it will be generated from `field_schema`.

    """

    name: str = "entity_extraction"
    input_template: str = (
        'Extract entities from the input text.\n\nInput:\n"""\n{text}\n"""'
    )
    labels: Optional[List[str]] = None
    output_template: str = "Extracted entities: {entities}"

    _quote_string_field_name: str = "quote_string"
    _default_quote_string_description: str = (
        "The text of the entity extracted from the input document."
    )
    _labels_field_name: str = "label"
    _default_labels_description: str = "The label assigned to the entity."

    def _get_input_field_name(self):
        # if field_schema is not provided, parse output_template and validate labels
        input_fields = self.get_input_fields()
        if len(input_fields) != 1:
            logger.warning(
                f"EntityExtraction skill only supports one input field, got {input_fields}. "
                f"Using the first one: {input_fields[0]}"
            )
        output_field_name = input_fields[0]
        return output_field_name

    def _get_output_field_name(self):
        # if field_schema is not provided, parse output_template and validate labels
        output_fields = self.get_output_fields()
        if len(output_fields) != 1:
            raise ValueError(
                f"EntityExtraction skill only supports one output field, got {output_fields}"
            )
        output_field_name = output_fields[0]
        return output_field_name

    @model_validator(mode="after")
    def validate_response_model(self):
        if self.response_model:
            raise NotImplementedError(
                "EntityExtraction skill does not support custom response model yet."
            )

        if self.field_schema:
            # in case field_schema is already provided, we don't need to parse output template and validate labels
            # check schema structure: it must contain one or two fields covering extracted quote string and labels
            validate_schema(self.field_schema)
            schema = next(iter(self.field_schema.values()))

            # identify quote string field as one field of "type": "string" without "enum"
            self._quote_string_field_name = next(
                (
                    k
                    for k, v in schema["items"]["properties"].items()
                    if v["type"] == "string" and "enum" not in v
                ),
                None,
            )
            if not self._quote_string_field_name:
                raise ValueError(
                    f"EntityExtraction skill output field items properties must contain one field "
                    f"of type string without enum (quote string), got {schema['items']['properties']}"
                )

            # check if description provided for quote string field, if not - warning and generate a default one
            if (
                "description"
                not in schema["items"]["properties"][self._quote_string_field_name]
            ):
                logger.warning(
                    f"EntityExtraction skill output field items properties quote string field must have 'description', "
                    f"generated default description: {self._default_quote_string_description}"
                )
                schema["items"]["properties"][self._quote_string_field_name][
                    "description"
                ] = self._default_quote_string_description

            if len(schema["items"]["properties"]) == 2:
                # identify labels list field as one field of "type": "string" with "enum"
                self._labels_field_name = next(
                    (
                        k
                        for k, v in schema["items"]["properties"].items()
                        if v["type"] == "string" and "enum" in v and v["enum"]
                    ),
                    None,
                )
                if not self._labels_field_name:
                    raise ValueError(
                        f"EntityExtraction skill output field items properties must have one field"
                        f" of type string with enum (list of labels), got {schema['items']['properties']}"
                    )
                # check if description provided for labels field, if not - warning and generate a default one
                if (
                    "description"
                    not in schema["items"]["properties"][self._labels_field_name]
                ):
                    logger.warning(
                        f"EntityExtraction skill output field items properties labels field must have 'description', "
                        f"generated default description: {self._default_labels_description}"
                    )
                    schema["items"]["properties"][self._labels_field_name][
                        "description"
                    ] = self._default_labels_description

        else:
            output_field_name = self._get_output_field_name()
            self.field_schema = {
                output_field_name: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            self._quote_string_field_name: {
                                "type": "string",
                                "description": self._default_quote_string_description,
                            }
                        },
                    },
                }
            }
            if self.labels:
                self.field_schema[output_field_name]["items"]["properties"][
                    self._labels_field_name
                ] = {
                    "type": "string",
                    "description": self._default_labels_description,
                    "enum": self.labels,
                }

        self.response_model = field_schema_to_pydantic_class(
            self.field_schema, self.name, self.description
        )
        return self

    def extract_indices(self, df):
        """
        Give the input dataframe with "text" column and "entities" column of the format
        ```
        [{"quote_string": "entity_1"}, {"quote_string": "entity_2"}, ...]
        ```
         extract the indices of the entities in the input text and put indices in the "entities" column:
         ```
         [{"quote_string": "entity_1", "start": 0, "end": 5}, {"quote_string": "entity_2", "start": 10, "end": 15}, ...]
         ```
        """
        input_field_name = self._get_input_field_name()
        output_field_name = self._get_output_field_name()
        df = extract_indices(
            df,
            input_field_name,
            output_field_name,
            self._quote_string_field_name,
            self._labels_field_name,
        )
        return df

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        output = super().apply(input, runtime)
        output = self.extract_indices(pd.concat([input, output], axis=1))
        return output

    async def aapply(
        self,
        input: InternalDataFrame,
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:
        output = await super().aapply(input, runtime)
        output = self.extract_indices(pd.concat([input, output], axis=1))
        return output
