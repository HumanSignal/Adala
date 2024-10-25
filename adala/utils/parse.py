import re
import string
import logging
from string import Formatter
from typing import List, TypedDict

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

    for match in re.finditer(r"(?<!\{)\{(.*?)\}(?!})", string):
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
