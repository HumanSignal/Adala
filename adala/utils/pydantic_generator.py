from typing import Any, Dict, List, Optional, Type, Union, Tuple
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, create_model


def json_schema_to_model(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Converts a JSON schema to a Pydantic model.

    Args:
        json_schema: The JSON schema to convert.

    Example:
        >>> json_schema_to_model({
        ...     "type": "object",
        ...     "title": "Person",
        ...     "description": "A person object",
        ...     "properties": {
        ...         "name": {
        ...             "type": "string",
        ...             "description": "The person's name"
        ...         },
        ...         "age": {
        ...             "type": "integer",
        ...             "description": "The person's age"
        ...         },
        ...         'profession': {
        ...             'type': 'string',
        ...             'description': 'The person\'s profession',
        ...             'enum': ['engineer', 'doctor', 'teacher']
        ...         }
        ...     },
        ... })
        class Person(BaseModel):
            '''A person object'''
            name: str = Field(..., description="The person's name")
            age: int = Field(..., description="The person's age")
            profession: Items = Field(..., description="The person's profession")

    Returns:
        A Pydantic model.
    """

    assert json_schema.get("type") == "object", "Only object schemas are supported"

    # `title` is the model class name
    model_name = json_schema.get("title", "Model")

    # `description` is the model class docstring
    model_description = json_schema.get("description", "")

    fields_def = {}
    for name, prop in json_schema.get("properties", {}).items():
        fields_def[name] = json_schema_to_pydantic_field(prop)

    # Create the BaseModel class using create_model().
    model = create_model(model_name, **fields_def)

    # Set the model docstring.
    model.__doc__ = model_description

    return model


def json_schema_to_pydantic_field(json_schema: Dict[str, Any]) -> Tuple[Any, Field]:
    """
    Converts a JSON schema property to a Pydantic field definition.

    Args:
        name: The field name.
        json_schema: The JSON schema property.

    Returns:
        A Pydantic field definition.
    """

    # Get the field type.
    type_ = json_schema_to_pydantic_type(json_schema)

    field_params = {}

    # Get the field description.
    description = json_schema.get("description")
    if description:
        field_params["description"] = description

    # Get the field examples.
    examples = json_schema.get("examples")
    if examples:
        field_params["examples"] = examples

    # Create a Field object with the type and optional parameters.
    return type_, Field(..., **field_params)


def json_schema_to_pydantic_type(
    json_schema: Dict[str, Any], enum_class_name: str = "Labels"
) -> Any:
    """
    Converts a JSON schema type to a Pydantic type.

    Args:
        json_schema: The JSON schema to convert.
        enum_class_name: The name of the Enum class to generate (TODO: propagate this parameter top-level).

    Returns:
        A Pydantic type.
    """

    type_ = json_schema.get("type")

    if type_ == "string":
        if "format" in json_schema:
            format_ = json_schema["format"]
            if format_ == "date-time":
                return datetime
            else:
                raise NotImplementedError(f"Unsupported JSON schema format: {format_}")
        elif "enum" in json_schema:
            return Enum(
                enum_class_name, {item: item for item in json_schema["enum"]}, type=str
            )
        return str
    elif type_ == "integer":
        return int
    elif type_ == "number":
        return float
    elif type_ == "boolean":
        return bool
    elif type_ == "array":
        items_schema = json_schema.get("items")
        if items_schema:
            item_type = json_schema_to_pydantic_type(items_schema)
            return List[item_type]
        else:
            return List
    elif type_ == "object":
        # Handle nested models.
        properties = json_schema.get("properties")
        if properties:
            nested_model = json_schema_to_model(json_schema)
            return nested_model
        else:
            return Dict
    elif type_ == "null":
        return Optional[Any]  # Use Optional[Any] for nullable fields
    else:
        raise ValueError(f"Unsupported JSON schema type: {type_}")
