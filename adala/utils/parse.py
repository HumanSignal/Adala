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
    TypedDict,
    Union,
)

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


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


match_fields_regex = re.compile(r"(?<!\{)\{([a-zA-Z0-9_]+)\}(?!})")


class MessageChunkType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_URLS = "image_urls"


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


# TODO: consolidate these data models and unify our preprocessing for LLM input into one step RawInputModel -> PreparedInputModel
class TextMessageChunk(TypedDict):
    type: Literal["text"]
    text: str


class ImageMessageChunk(TypedDict):
    type: Literal["image"]
    image_url: Dict[str, str]


MessageChunk = Union[TextMessageChunk, ImageMessageChunk]

Message = Union[str, List[MessageChunk]]


def split_message_into_chunks(
    input_template: str, input_field_types: Dict[str, MessageChunkType], **payload
) -> List[MessageChunk]:
    """Split a template string into message chunks based on field types.

    Args:
        input_template: Template string with placeholders like '{field_name}'
        input_field_types: Mapping of field names to their chunk types
        payload: Dictionary with values to substitute into the template instead of placeholders

    Returns:
        List of message chunks with appropriate type and content:
        - Text chunks: {'type': 'text', 'text': str}
        - Image chunks: {'type': 'image_url', 'image_url': {'url': str}}

    Example:
        >>> split_message_into_chunks(
        ...     'Look at {image} and describe {text}',
        ...     {'image': MessageChunkType.IMAGE_URL, 'text': MessageChunkType.TEXT},
        ...     {'image': 'http://example.com/img.jpg', 'text': 'this content'}
        ... )
        [
            {'type': 'text', 'text': 'Look at '},
            {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.jpg'}},
            {'type': 'text', 'text': ' and describe this content'}
        ]
    """
    # Parse template to get chunks with field positions and types
    parsed = parse_template(
        input_template,
        include_texts=True,
        payload=payload,
        input_field_types=input_field_types,
    )

    logger.debug(f"Parsed template: {parsed}")

    result = []
    current_text = ""

    def _add_current_text_as_chunk():
        # this function is used to flush `current_text` buffer into a text chunk, and start over
        nonlocal current_text
        if current_text:
            result.append({"type": "text", "text": current_text})
            current_text = ""

    for part in parsed:
        # iterate over parsed chunks - they already contains field types and placeholder values
        if part["type"] == "text":
            # each text chunk without placeholders is added to the current buffer and we continue
            current_text += part["text"]
        elif part["type"] == "var":
            field_type = part["field_type"]
            field_value = part["data"]
            if field_value is None:
                # if field value is not provided, it is assumed to be a text field
                current_text += part["text"]
            else:
                match field_type:
                    case MessageChunkType.TEXT:
                        # For text fields, we don't break chunks and add text fields to current buffer
                        current_text += (
                            str(field_value)
                            if field_value is not None
                            else part["text"]
                        )

                    case MessageChunkType.IMAGE_URL:
                        # Add remaining text as text chunk
                        _add_current_text_as_chunk()
                        # Add image URL as new image chunk
                        result.append(
                            {"type": "image_url", "image_url": {"url": field_value}}
                        )

                    case MessageChunkType.IMAGE_URLS:
                        assert isinstance(
                            field_value, List
                        ), "Image URLs must be a list"
                        # Add remaining text as text chunk
                        _add_current_text_as_chunk()
                        # Add image URLs as new image chunks
                        for url in field_value:
                            result.append(
                                {"type": "image_url", "image_url": {"url": url}}
                            )

                    case _:
                        # Handle unknown field types as text
                        current_text += part["text"]

    # Add any remaining text
    _add_current_text_as_chunk()

    logger.debug(f"Result: {result}")

    return result


class MessagesBuilder(BaseModel):
    user_prompt_template: str
    system_prompt: Optional[str] = None
    instruction_first: bool = True
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    split_into_chunks: bool = False
    input_field_types: DefaultDict[
        str,
        Annotated[
            MessageChunkType, Field(default_factory=lambda: MessageChunkType.TEXT)
        ],
    ] = Field(default_factory=lambda: defaultdict(lambda: MessageChunkType.TEXT))

    def get_messages(self, payload: Dict[str, Any]):
        if self.split_into_chunks:
            user_prompt = split_message_into_chunks(
                input_template=self.user_prompt_template,
                input_field_types=self.input_field_types,
                **payload,
                **self.extra_fields,
            )
        else:
            user_prompt = partial_str_format(
                self.user_prompt_template, **payload, **self.extra_fields
            )
        messages = [{"role": "user", "content": user_prompt}]
        if self.system_prompt:
            if self.instruction_first:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            else:
                messages[0]["content"] += self.system_prompt
        return messages
