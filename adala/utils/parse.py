import re
import string
import logging
from typing import (
    List,
    TypedDict,
    Dict,
    Optional,
    Union,
    Literal,
    Iterable,
    Generator,
    Any,
)
from enum import Enum
from pydantic import BaseModel, Field
from collections import defaultdict

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


def parse_template(string, include_texts=True) -> List[TemplateChunks]:
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

    for match in match_fields_regex.finditer(string):
        # for match in re.finditer(r'\{(.*?)\}', string):
        # Text before field
        if last_index < match.start() and include_texts:
            text = string[last_index : match.start()]
            chunks.append(
                {
                    "text": text,
                    "start": last_index,
                    "end": match.start(),
                    "type": "text",
                }
            )

        # Field itself
        field = match.group(1)
        start = match.start()
        end = match.end()
        chunks.append({"text": field, "start": start, "end": end, "type": "var"})

        last_index = match.end()

    # Text after the last field
    if last_index < len(string) and include_texts:
        text = string[last_index:]
        chunks.append(
            {"text": text, "start": last_index, "end": len(string), "type": "text"}
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


class MessageChunkType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"


def split_message_into_chunks(
    input_template: str, input_field_types: Dict[str, MessageChunkType], **input_fields
) -> List[MessageChunk]:
    """Split a template string with field types into a list of message chunks.

    Takes a template string with placeholders and splits it into chunks based on the field types,
    preserving the text between placeholders.

    Args:
        input_template (str): Template string with placeholders, e.g. '{a} is a {b} is an {a}'
        input_field_types (Dict[str, MessageChunkType]): Dict mapping field names to their types
        **input_fields: Field values to substitute into template

    Returns:
        List[Dict[str, str]]: List of message chunks with appropriate type and content.
            Text chunks have format: {'type': 'text', 'text': str}
            Image chunks have format: {'type': 'image_url', 'image_url': {'url': str}}

    Example:
        >>> split_message_into_chunks(
        ...     '{a} is a {b} is an {a}',
        ...     {'a': MessageChunkType.TEXT, 'b': MessageChunkType.IMAGE_URL},
        ...     a='the letter a',
        ...     b='http://example.com/b.jpg'
        ... )
        [
            {'type': 'text', 'text': 'the letter a is a '},
            {'type': 'image_url', 'image_url': {'url': 'http://example.com/b.jpg'}},
            {'type': 'text', 'text': ' is an the letter a'}
        ]
    """
    # Parse template to get field positions and surrounding text
    parsed = parse_template(input_template)

    def add_to_current_chunk(
        current_chunk: Optional[MessageChunk], chunk: MessageChunk
    ) -> MessageChunk:
        if current_chunk:
            current_chunk["text"] += chunk["text"]
            return current_chunk
        else:
            return chunk

    # Build chunks by iterating through parsed template parts
    def build_chunks(
        parsed: Iterable[TemplateChunks],
    ) -> Generator[MessageChunk, None, None]:
        current_chunk: Optional[MessageChunk] = None

        for part in parsed:
            if part["type"] == "text":
                current_chunk = add_to_current_chunk(
                    current_chunk, {"type": "text", "text": part["text"]}
                )
            elif part["type"] == "var":
                field_value = part["text"]
                try:
                    field_type = input_field_types[field_value]
                except KeyError:
                    raise ValueError(
                        f"Field {field_value} not found in input_field_types. Found fields: {input_field_types}"
                    )
                if field_type == MessageChunkType.TEXT:
                    # try to substitute in variable and add to current chunk
                    substituted_text = partial_str_format(
                        f"{{{field_value}}}", **input_fields
                    )
                    if substituted_text != field_value:
                        current_chunk = add_to_current_chunk(
                            current_chunk, {"type": "text", "text": substituted_text}
                        )
                    else:
                        # be permissive for unfound variables
                        current_chunk = add_to_current_chunk(
                            current_chunk,
                            {"type": "text", "text": f"{{{field_value}}}"},
                        )
                elif field_type == MessageChunkType.IMAGE_URL:
                    substituted_text = partial_str_format(
                        f"{{{field_value}}}", **input_fields
                    )
                    if substituted_text != field_value:
                        # push current chunk, push image chunk, and start new chunk
                        if current_chunk:
                            yield current_chunk
                        current_chunk = None
                        yield {
                            "type": "image_url",
                            "image_url": {"url": input_fields[field_value]},
                        }
                    else:
                        # be permissive for unfound variables
                        current_chunk = add_to_current_chunk(
                            current_chunk,
                            {"type": "text", "text": f"{{{field_value}}}"},
                        )

        if current_chunk:
            yield current_chunk

    return list(build_chunks(parsed))


class MessagesBuilder(BaseModel):
    user_prompt_template: str
    system_prompt: Optional[str] = None
    instruction_first: bool = True
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    split_into_chunks: bool = False
    input_field_types: DefaultDict[str, Annotated[MessageChunkType, Field(default_factory=lambda: MessageChunkType.TEXT)]] = Field(default_factory=lambda: defaultdict(lambda: MessageChunkType.TEXT))


    def get_messages(self, payload: Dict[str, Any]):
        if self.split_into_chunks:
            user_prompt = split_message_into_chunks(
                self.user_prompt_template,
                self.input_field_types,
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
