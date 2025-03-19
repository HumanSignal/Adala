"""
Label Studio Processor

A simplified interface for Label Studio integration with Adala.
This module provides functionality to process data through Label Studio's annotation schema
using LLMs for automated labeling.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Union, Type, Sequence
from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from pydantic import BaseModel

from adala.core.processor import DataProcessor
from adala.utils.internal_data import DataTable
from adala.runtimes.batch_llm import BatchLLMRuntime

# Import Label Studio SDK
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk._extensions.label_studio_tools.core.utils.json_schema import (
    json_schema_to_pydantic,
)

logger = logging.getLogger(__name__)


@dataclass
class NERTag:
    """Represents a Named Entity Recognition tag with its properties."""

    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None
    labels: List[str] = None

    def __post_init__(self):
        self.labels = self.labels or []


def extract_variable_name(input_string: str) -> List[str]:
    """Extract variable names specified as $<variable-name> from a string."""
    pattern = r"\$([a-zA-Z0-9_]+)"
    return re.findall(pattern, input_string)


def validate_ner_tag_format(
    df: DataTable, input_field: str, output_field: str
) -> DataTable:
    """
    Validate and fix NER tag format in the output.

    Args:
        df: Input DataTable containing both input and output fields
        input_field: Name of the input text field
        output_field: Name of the output field containing NER tags

    Returns:
        DataTable with validated and fixed NER tag format
    """
    for _, row in df.iterrows():
        if row.get("_error"):
            continue

        input_text = row.get(input_field, "")
        output_field_data = row.get(output_field)

        if not isinstance(output_field_data, list):
            continue

        for item in output_field_data:
            if not isinstance(item, dict):
                continue

            start = item.get("start")
            end = item.get("end")

            if (
                start is not None
                and end is not None
                and 0 <= start < end <= len(input_text)
            ):
                if not item.get("text"):
                    item["text"] = input_text[start:end]

    return df


def extract_missing_indices(
    df: DataTable, input_field: str, output_field: str, text_field: str = "text"
) -> DataTable:
    """
    Extract missing start and end indices for NER tags.

    Args:
        df: Input DataTable containing both input and output fields
        input_field: Name of the input text field
        output_field: Name of the output field containing NER tags
        text_field: Name of the field containing the text to match

    Returns:
        DataTable with extracted indices where missing
    """
    for _, row in df.iterrows():
        if row.get("_error"):
            continue

        input_text = row.get(input_field, "")
        output_field_data = row.get(output_field)

        if not isinstance(output_field_data, list):
            continue

        for item in output_field_data:
            if not isinstance(item, dict):
                continue

            if (item.get("start") is None or item.get("end") is None) and item.get(
                text_field
            ):
                text = item.get(text_field)
                if isinstance(text, str):
                    start = input_text.find(text)
                    if start >= 0:
                        item["start"] = start
                        item["end"] = start + len(text)

    return df


class LabelStudioProcessor(DataProcessor):
    """
    A processor for Label Studio annotations that uses LLMs for automated labeling.

    This processor takes a Label Studio XML configuration and generates appropriate
    prompts and response models for annotation tasks.
    """

    def __init__(
        self,
        label_config: str,
        allowed_control_tags: Optional[Sequence[str]] = None,
        allowed_object_tags: Optional[Sequence[str]] = None,
        instructions: Optional[str] = None,
        runtime: Optional[BatchLLMRuntime] = None,
        **runtime_kwargs,
    ):
        """
        Initialize a Label Studio processor.

        Args:
            label_config: Label Studio XML configuration
            allowed_control_tags: Optional sequence of control tag names to include
            allowed_object_tags: Optional sequence of object tag names to include
            instructions: Custom instructions for the LLM
            runtime: BatchLLMRuntime instance (created automatically if not provided)
            runtime_kwargs: Additional arguments to pass to BatchLLMRuntime if created
        """
        self.label_config = label_config
        self.allowed_control_tags = allowed_control_tags
        self.allowed_object_tags = allowed_object_tags

        # Create and configure label interface
        self._label_interface = self._create_filtered_interface()

        # Get schema from interface
        self.field_schema = self._label_interface.to_json_schema()

        # Initialize the processor
        super().__init__(
            prompt_template=instructions
            or "Annotate the input data according to the provided schema.",
            response_model=BaseModel,  # Placeholder, will be set dynamically
            runtime=runtime,
            **runtime_kwargs,
        )

    def _create_filtered_interface(self) -> LabelInterface:
        """Create a filtered LabelInterface based on allowed tags."""
        if not (self.allowed_control_tags or self.allowed_object_tags):
            return LabelInterface(self.label_config)

        control_tags = {
            tag: self._label_interface._controls[tag]
            for tag in (self.allowed_control_tags or self._label_interface._controls)
        }

        object_tags = {
            tag: self._label_interface._objects[tag]
            for tag in (self.allowed_object_tags or self._label_interface._objects)
        }

        interface = LabelInterface.create_instance(tags={**control_tags, **object_tags})
        logger.debug(
            f"Filtered labeling config based on allowed tags: {interface.config}"
        )
        return interface

    @property
    def ner_tags(self) -> List[ControlTag]:
        """Get NER tags from the label config."""
        control_tag_names = self.allowed_control_tags or list(
            self._label_interface._controls.keys()
        )
        return [
            tag
            for tag_name in control_tag_names
            if (tag := self._label_interface.get_control(tag_name)).tag.lower()
            in {"labels", "hypertextlabels"}
            and (
                not self.allowed_object_tags
                or all(
                    object_tag.tag in self.allowed_object_tags
                    for object_tag in tag.objects
                )
            )
        ]

    @property
    def image_tags(self) -> List[ObjectTag]:
        """Get image tags from the label config."""
        object_tag_names = self.allowed_object_tags or list(
            self._label_interface._objects.keys()
        )
        return [
            tag
            for tag_name in object_tag_names
            if (tag := self._label_interface.get_object(tag_name)).tag.lower()
            == "image"
        ]

    def _process_ner_tags(self, df: DataTable, result: DataTable) -> DataTable:
        """Process NER tags in the result DataTable."""
        for ner_tag in self.ner_tags:
            if not ner_tag.objects:
                continue

            input_field = ner_tag.objects[0].value.lstrip("$")
            output_field = ner_tag.name

            # Join input and output data
            combined_df = pd.concat([df, result], axis=1)

            # Validate and fix NER output format
            result = validate_ner_tag_format(combined_df, input_field, output_field)

            # Extract indices if missing
            result = extract_missing_indices(result, input_field, output_field)

        return result

    def process(
        self,
        data: Union[pd.DataFrame, List[Dict], DataTable],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> DataTable:
        """Process a batch of data through the Label Studio processor."""
        # Convert input to DataTable
        df = (
            DataTable(data)
            if isinstance(data, list)
            else (
                DataTable.from_dataframe(data)
                if isinstance(data, pd.DataFrame) and not isinstance(data, DataTable)
                else data
            )
        )

        # Get dynamic response model from schema
        with json_schema_to_pydantic(self.field_schema) as ResponseModel:
            self.response_model = ResponseModel
            result = super().process(data, extra_context)
            return self._process_ner_tags(df, result)

    async def aprocess(
        self,
        data: Union[pd.DataFrame, List[Dict], DataTable],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> DataTable:
        """Process a batch of data through the Label Studio processor asynchronously."""
        # Convert input to DataTable
        df = (
            DataTable(data)
            if isinstance(data, list)
            else (
                DataTable.from_dataframe(data)
                if isinstance(data, pd.DataFrame) and not isinstance(data, DataTable)
                else data
            )
        )

        # Get dynamic response model from schema
        with json_schema_to_pydantic(self.field_schema) as ResponseModel:
            self.response_model = ResponseModel
            result = await super().aprocess(data, extra_context)
            return self._process_ner_tags(df, result)
