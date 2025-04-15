import logging
import re
import string
from collections import defaultdict
from enum import Enum
from typing import (
    Annotated,
    Any,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

from litellm.utils import token_counter
import litellm


class PartialStringFormatter(string.Formatter):
    def __init__(self):
        super().__init__()
        self._current_field_name = None

    def get_field(self, field_name, args, kwargs):
        self._current_field_name = field_name
        try:
            return super().get_field(field_name, args, kwargs)
        except (KeyError, AttributeError):
            # For unprovided variables, preserve the entire field name including format spec
            return "{" + field_name + "}", field_name

    def format_field(self, value, format_spec):
        if isinstance(value, str) and value.startswith("{"):
            # This is a preserved placeholder, return as is
            if format_spec:
                return value[:-1] + ":" + format_spec + "}"
            else:
                return value

        try:
            return super().format_field(value, format_spec)
        except (ValueError, TypeError):
            # If format spec is invalid, preserve the original field name and format spec
            if format_spec:
                return "{" + self._current_field_name + ":" + format_spec + "}"
            return str(value)


PartialStringFormat = PartialStringFormatter()


def partial_str_format(format_string: str, **kwargs) -> str:
    """
    Format a string with the provided variables while preserving any unprovided placeholders.
    Preserves format specifiers for both provided and unprovided variables.

    Args:
        format_string: The string to format
        **kwargs: The variables to use for formatting

    Returns:
        The formatted string with preserved unprovided placeholders and format specifiers

    Examples:
        >>> partial_str_format("Hello {name}!", name="World")
        'Hello World!'
        >>> partial_str_format("Hello {name} {unknown}!", name="World")
        'Hello World {unknown}!'
        >>> partial_str_format("Value: {x:.2f}", x="not_a_float")
        'Value: {x:.2f}'
    """
    if not format_string:
        return ""

    # Temporarily replace valid format strings to protect them from escaping
    format_pattern = re.compile(r"\{[^{}]+\}")
    markers = {
        f"__MARKER_{i}__": m.group(0)
        for i, m in enumerate(format_pattern.finditer(format_string))
    }

    processed = format_string
    for marker, format_str in markers.items():
        processed = processed.replace(format_str, marker)

    # Escape remaining brackets and restore format strings
    processed = processed.replace("{", "{{").replace("}", "}}")
    for marker, format_str in markers.items():
        processed = processed.replace(marker, format_str)

    return PartialStringFormat.format(processed, **kwargs)


class TemplateChunks(TypedDict):
    text: str
    start: int
    end: int
    type: str
    data: Optional[Any]
    field_type: Optional[Any]


match_fields_regex = re.compile(r"(?<!\{)\{([a-zA-Z0-9_]+)\}(?!})")


class MessageChunkType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_URLS = "image_urls"


class TextMessageChunk(TypedDict):
    type: Literal["text"]
    text: str


class ImageMessageChunk(TypedDict):
    type: Literal["image"]
    image_url: Dict[str, str]


MessageChunk = Union[TextMessageChunk, ImageMessageChunk]

Message = Union[str, List[MessageChunk]]


def parse_template(
    string,
    include_texts=True,
    payload: Optional[Dict[str, Any]] = None,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
) -> List[TemplateChunks]:
    """
    Parses a template string to extract output fields and the text between them.

    Args:
        string (str): The template string to parse.
        include_texts (bool): Whether to include the text between the fields in the output.

    Returns:
        List[Element]: A list of dictionaries with the keys 'text', 'start', 'end', and 'type'.

    Example:
        >>> parse_template("some text {field1} some more text {field2}")
        [{"text": "some text ", "start": 0, "end": 10, "type": "text"},
         {"text": "field1", "start": 11, "end": 17, "type": "var"},
         {"text": " some more text ", "start": 18, "end": 35, "type": "text"},
         {"text": "field2", "start": 36, "end": 42, "type": "var"}]
    """

    chunks: List[TemplateChunks] = []
    last_index = 0
    payload = payload or {}
    input_field_types = input_field_types or {}

    for match in match_fields_regex.finditer(string):
        # for match in re.finditer(r'\{(.*?)\}', string):
        # Text before field
        start = match.start()
        if last_index < start and include_texts:
            text = string[last_index:start]
            chunks.append(
                {
                    "text": text,
                    "start": last_index,
                    "end": start,
                    "type": "text",
                    "data": None,
                    "field_type": None,
                }
            )

        # Field itself
        end = match.end()
        field = string[start:end].strip("{}")
        # Extract the field name by removing the brackets
        data = payload.get(field)
        field_type = input_field_types.get(field, MessageChunkType.TEXT)
        chunks.append(
            {
                "text": field,
                "start": start,
                "end": end,
                "type": "var",
                "data": data,
                "field_type": field_type,
            }
        )

        last_index = end

    # Text after the last field
    if last_index < len(string) and include_texts:
        text = string[last_index:]
        chunks.append(
            {
                "text": text,
                "start": last_index,
                "end": len(string),
                "type": "text",
                "data": None,
                "field_type": None,
            }
        )

    return chunks
