import re
import string
from string import Formatter
from typing import List, TypedDict, Dict, Any, Type, Optional
from adala.utils.pydantic_generator import json_schema_to_model
from pydantic import BaseModel, Field, create_model


class PartialStringFormatter(string.Formatter):
    def get_value(self, key, args, kwds):
        if isinstance(key, str):
            try:
                return kwds[key]
            except KeyError:
                return "{" + key + "}"
        else:
            Formatter.get_value(key, args, kwds)


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


def parse_template_to_pydantic_class(
    output_template: str,
    provided_field_schema: Optional[Dict[str, Any]] = None,
    class_name: str = 'Output',
    description: str = ''
) -> Type[BaseModel]:
    """
    Parses a template string to extract output fields and map them to the pydantic BaseModel class definition.
    Variable prefixes with stripped punctuation will replace `description` fields in the schema if not provided.
    For example:
        "Model output: {field1}" with schema {"field1": {"type": "string"}} will become:
        ```python
        class Template(BaseModel):
            field1: str = Field(..., description="Model output")
        ```

    Args:
        output_template (str): The template string to parse, for example "Model output: {field1}".
        provided_field_schema (Dict[str, Any]): The standard JSON schema of the fields. For example:
        ```json
        {
            "field1": {
                "type": "string",
                "description": "Model output"
            }
        }
        ```
        class_name (str): The name of the Pydantic class to generate.
        description (str): The docstring of the Pydantic class

    Returns:
        Type[BaseModel]: A Pydantic class representing the template.

    Example:
        >>> parse_template_to_pydantic_class("some text {field1} some more labels {field2}", {"field1": {"type": "string"}, "field2": {"type": "array", "items": {"type": "string", "enum": ["label1", "label2"]}}}, "Template", "A template class")
        class Template(BaseModel):
            '''A template class'''
            field1: str = Field(..., description="some text")
            field2: List[Items] = Field(..., description="some more labels")
    """

    chunks = parse_template(output_template)

    provided_field_schema = provided_field_schema or {}
    properties = {}
    previous_text = ''
    for chunk in chunks:
        if chunk["type"] == "text":
            previous_text = chunk["text"]
        if chunk["type"] == "var":
            field_name = chunk["text"]
            field_schema_entry = provided_field_schema.get(field_name, {})
            # by default, all fields are strings
            field_type = field_schema_entry.get("type", "string")

            # if description is not provided, use the text before the field,
            # otherwise use the field name with underscores replaced by spaces
            field_description = field_schema_entry.get(
                "description", previous_text) or field_name.replace("_", " ")
            field_description = field_description.strip(string.punctuation).strip()
            previous_text = ''

            # create JSON schema entry for the field
            properties[field_name] = {
                "type": field_type,
                "description": field_description
            }
            # add the rest of the fields from provided_field_schema
            for key, value in field_schema_entry.items():
                if key not in properties[field_name]:
                    properties[field_name][key] = value

    json_schema = {
        "type": "object",
        "title": class_name,
        "description": description,
        "properties": properties
    }

    return json_schema_to_model(json_schema)
