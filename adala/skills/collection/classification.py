import logging
import jsonschema
from jsonschema import validate

from adala.skills._base import Skill, TransformSkill
from adala.utils.pydantic_generator import field_schema_to_pydantic_class
from typing import List, Dict, Optional, Any
from pydantic import model_validator

logger = logging.getLogger(__name__)


def validate_schema(schema: Dict[str, Any]):
    
    single_label_schema = {
        "type": "object",
        "patternProperties": {
            ".*": {
                "type": "object",
                "properties": {
                        "type": {"type": "string", "enum": ["string"]},
                        "enum": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "description": {"type": "string"},
                },
                "required": ["type", "enum"],
                "additionalProperties": False,
            },
        },
        "minProperties": 1,
        "maxProperties": 1,
        "additionalProperties": False,
    }
    
    multi_label_schema = {
        "type": "object",
        "patternProperties": {
            ".*": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["array"]},
                    "items": {
                        "type": "object",
                        "properties": {
                            # label definition has the same format as single label
                            "type": {"type": "string", "enum": ["string"]},
                            "enum": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": ["type", "enum"],
                        "additionalProperties": False,
                    },
                    "uniqueItems": {"type": "boolean", "default": True},
                    "minItems": {"type": "integer", "default": 1},
                    "maxItems": {"type": "integer"},
                    "description": {"type": "string"},
                },
            "required": ["type", "items"],
            "additionalProperties": False,
            },
        },
        "minProperties": 1,
        "maxProperties": 1,
        "additionalProperties": False,
    }

    expected_schema = {
        "type": "object",
        "oneOf": [single_label_schema, multi_label_schema]
    }

    try:
        validate(instance=schema, schema=expected_schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ValueError(f"Invalid schema: {e.message}")


class ClassificationSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    Classification skill uses labels to classify input text that are provided in one of the ways:
    - as a field schema, for example:
    ```
    {
        "output_field": {
            "type": "string",
            "enum": ["label_1", "label_2", "label_3"]
        }
    }
    ```
    - as a parameter in the skill configuration, for example:
    ClassificationSkill(output_template="{my_output}", labels={"my_output": ["label_1", "label_2", "label_3"]})
    """

    name: str = "classification"
    instructions: str = "Classify input text."
    input_template: str = 'Input:\n"""\n{text}\n"""'
    output_template: str = "Classification result: {label}"
    labels: Optional[List[str]] = None

    _default_output_field_description: str = "The classification label"

    @model_validator(mode="after")
    def validate_response_model(self):

        if self.response_model:
            # TODO validate schema above against response_model.schema()
            '''
            Example schema generated for multilabel:

                {'$defs': {'SupportTag': {'enum': ['Account Access',
                    'Login Issues',
                    'App Functionality',
                    'Bug Report',
                    'Billing',
                    'Account Management',
                    'User Settings',
                    'Notifications',
                    'Performance',
                    'Website Issues'],
                   'title': 'SupportTag',
                   'type': 'string'}},
                 'properties': {'predicted_tags': {'items': {'$ref': '#/$defs/SupportTag'},
                   'minItems': 1,
                   'title': 'Predicted Tags',
                   'type': 'array',
                   'uniqueItems': True}},
                 'required': ['predicted_tags'],
                 'title': 'Output',
                 'type': 'object'}

            Example schema generated for single label:

                {'properties': {'predicted_category': {'description': 'The classification label',
                   'enum': ['Footwear/Clothing',
                    'Electronics',
                    'Food/Beverages',
                    'Furniture/Home Decor',
                    'Beauty/Personal Care'],
                   'title': 'Predicted Category',
                   'type': 'string'}},
                 'required': ['predicted_category'],
                 'title': 'Output',
                 'type': 'object'}

            this doesn't pass validate_schema(), so models and schemas don't roundtrip correctly.
            '''
            return self

        if self.field_schema:
            # in case field_schema is already provided, we don't need to parse output template and validate labels
            # just ensure that schema contains exactly one field and enum is presented
            validate_schema(self.field_schema)
            schema = list(self.field_schema.values())[0]

            # check if description is provided for the output field, if not - warning and generate a default one
            if "description" not in schema:
                logger.warning(
                    f"Classification skill output field must have 'description', "
                    f"generated default description: {self._default_output_field_description}"
                )
                schema["description"] = self._default_output_field_description

        else:
            # if field_schema is not provided, parse output_template and validate labels
            self.field_schema = {}
            output_fields = self.get_output_fields()
            if len(output_fields) != 1:
                raise ValueError(
                    f"Classification skill only supports one output field, got {output_fields}"
                )
            output_field_name = output_fields[0]
            if not self.labels:
                raise ValueError(
                    "Classification skill requires labels to be provided either as a field schema or as a parameter"
                )

            self.field_schema[output_field_name] = {
                "type": "string",
                "description": self._default_output_field_description,
                "enum": self.labels,
            }

        self.response_model = field_schema_to_pydantic_class(
            self.field_schema, self.name, self.description
        )
        return self
