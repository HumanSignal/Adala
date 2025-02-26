import re
import string
import logging
from string import Formatter
from typing import List, TypedDict, Dict, Optional, Union, Literal, Iterable, Generator
from enum import Enum

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