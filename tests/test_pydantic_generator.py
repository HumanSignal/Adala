import pytest
import string
from pydantic import BaseModel, Field
from typing import List, Literal
from adala.utils.pydantic_generator import (
    json_schema_to_model,
    field_schema_to_pydantic_class,
)

# Sample JSON schema
sample_schema = {
    "type": "object",
    "title": "Person",
    "description": "A person object",
    "properties": {
        "name": {"type": "string", "description": "The person's name"},
        "age": {"type": "integer", "description": "The person's age"},
        "profession": {
            "type": "string",
            "description": "The person's profession",
            "enum": ["engineer", "doctor", "teacher"],
        },
    },
}


# Expected Pydantic model
class Person(BaseModel):
    """A person object"""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    profession: Literal["engineer", "doctor", "teacher"] = Field(
        ..., description="The person's profession"
    )


def test_json_schema_to_model():
    # Convert JSON schema to Pydantic model
    GeneratedModel = json_schema_to_model(sample_schema)

    # Assert that the generated model matches the expected model
    assert GeneratedModel.__name__ == Person.__name__
    assert GeneratedModel.__doc__ == Person.__doc__

    # Create an instance of the generated model
    instance = GeneratedModel(name="John Doe", age=30, profession="engineer")

    # Assert the instance fields
    assert instance.name == "John Doe"
    assert instance.age == 30

    # Create an instance of the expected model
    expected_instance = Person(name="John Doe", age=30, profession="engineer")

    # Assert that the instances are equivalent
    assert instance.name == expected_instance.name
    assert instance.age == expected_instance.age
    assert instance.profession == expected_instance.profession


@pytest.mark.parametrize(
    "field_schema, fields_descriptions, good_params, bad_params, ",
    (
        # create from field schema
        (
            {
                "field1": {"type": "string", "description": "Description for field1"},
                "field2": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["label1", "label2"]},
                },
            },
            ("Description for field1", None),
            {"field1": "value1", "field2": ["label1"]},
            {"field1": "value1", "field2": ["label2", "non_existent_label"]},
        ),
    ),
)
def test_field_schema_to_pydantic_class(
    field_schema, fields_descriptions, good_params, bad_params
):

    # Generated Pydantic class
    GeneratedClassDef = field_schema_to_pydantic_class(
        field_schema, "MySuperClass", "A super class"
    )

    assert GeneratedClassDef.__name__ == "MySuperClass"
    assert GeneratedClassDef.__doc__ == "A super class"
    assert GeneratedClassDef.model_fields["field1"].annotation == str
    assert (
        GeneratedClassDef.model_fields["field1"].description == fields_descriptions[0]
    )
    assert (
        GeneratedClassDef.model_fields["field2"].description == fields_descriptions[1]
    )
    assert GeneratedClassDef.model_fields["field1"].is_required()
    assert GeneratedClassDef.model_fields["field2"].is_required()

    # Create an instance of the generated class
    instance = GeneratedClassDef(**good_params)
    assert instance.field1 == good_params["field1"]
    assert instance.field2 == good_params["field2"]

    # assert there is exception when trying to generate non-existent label for field2
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GeneratedClassDef(**bad_params)
