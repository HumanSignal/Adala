import re
import string
import logging
from string import Formatter
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
    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            try:
                return kwds[key]
            except KeyError:
                return "{" + key + "}"
        else:
            Formatter.get_value(key, args, kwds)

    def format_field(self, value, format_spec):
        try:
            return super().format_field(value, format_spec)
        except ValueError:
            # HACK: the value was an unfilled variable or not a variable at all, so the format spec should be considered part of the variable name
            if value.startswith("{") and value.endswith("}"):
                return value[:-1] + ":" + format_spec + "}"

    def _vformat(
        self, format_string, args, kwargs, used_args, recursion_depth, auto_arg_index=0
    ):
        # copied verbatim from parent class except for the # HACK
        if recursion_depth < 0:
            raise ValueError("Max string recursion exceeded")
        result = []
        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):

            # output the literal text
            if literal_text:
                result.append(literal_text)

            # if there's a field, output it
            if field_name is not None:
                # this is some markup, find the object and do
                #  the formatting

                # handle arg indexing when empty field_names are given.
                if field_name == "":
                    if auto_arg_index is False:
                        raise ValueError(
                            "cannot switch from manual field "
                            "specification to automatic field "
                            "numbering"
                        )
                    field_name = str(auto_arg_index)
                    auto_arg_index += 1
                elif field_name.isdigit():
                    if auto_arg_index:
                        raise ValueError(
                            "cannot switch from manual field "
                            "specification to automatic field "
                            "numbering"
                        )
                    # disable auto arg incrementing, if it gets
                    # used later on, then an exception will be raised
                    auto_arg_index = False

                # given the field_name, find the object it references
                #  and the argument it came from
                obj, arg_used = self.get_field(field_name, args, kwargs)
                used_args.add(arg_used)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)

                # expand the format spec, if needed
                format_spec, auto_arg_index = self._vformat(
                    format_spec,
                    args,
                    kwargs,
                    used_args,
                    recursion_depth - 1,
                    auto_arg_index=auto_arg_index,
                )

                # format the object and append to the result
                # HACK: if the format_spec is invalid, assume this field_name was not meant to be a variable, and don't substitute anything
                formatted_field = self.format_field(obj, format_spec)
                if formatted_field is None:
                    result.append("{" + ":".join([field_name, format_spec]) + "}")
                else:
                    result.append(formatted_field)

        return "".join(result), auto_arg_index


PartialStringFormat = PartialStringFormatter()


def partial_str_format(string, **kwargs):
    """
    Formats a string with a subset of the arguments.
    Analogous to str.format, but ignores missing arguments.
    """
    return PartialStringFormat.format(string, **kwargs)


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
        payload: Dictionary with values to substitute into the template

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
        nonlocal current_text
        if current_text:
            result.append({"type": "text", "text": current_text})
            current_text = ""

    for part in parsed:
        if part["type"] == "text":
            current_text += part["text"]
        elif part["type"] == "var":
            field_type = part["field_type"]
            field_value = part["data"]
            if field_value is None:
                current_text += part["text"]
            else:
                match field_type:
                    case MessageChunkType.TEXT:
                        # Add text fields to current buffer
                        current_text += (
                            str(field_value)
                            if field_value is not None
                            else part["text"]
                        )

                    case MessageChunkType.IMAGE_URL:
                        # Add remaining text as text chunk
                        _add_current_text_as_chunk()
                        # Add image URL as image chunk
                        result.append(
                            {"type": "image_url", "image_url": {"url": field_value}}
                        )

                    case MessageChunkType.IMAGE_URLS:
                        assert isinstance(
                            field_value, List
                        ), "Image URLs must be a list"
                        # Add remaining text as text chunk
                        _add_current_text_as_chunk()
                        # Add image URLs as image chunks
                        for url in field_value:
                            result.append(
                                {"type": "image_url", "image_url": {"url": url}}
                            )

                    case _:
                        # Handle unknown field types as text
                        current_text += part["text"]

    # Add any remaining text
    _add_current_text_as_chunk()

    return result


class MessagesBuilder(BaseModel):
    user_prompt_template: str
    system_prompt: Optional[str] = None
    instruction_first: bool = True
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    split_into_chunks: bool = False
    input_field_types: Optional[Dict[str, MessageChunkType]] = Field(default=None)

    def get_messages(self, payload: Dict[str, Any]):
        if self.split_into_chunks:
            input_field_types = self.input_field_types or defaultdict(
                lambda: MessageChunkType.TEXT
            )
            user_prompt = split_message_into_chunks(
                input_template=self.user_prompt_template,
                input_field_types=input_field_types,
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
