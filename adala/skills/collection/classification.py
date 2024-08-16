import logging

from adala.skills._base import Skill, TransformSkill
from adala.utils.pydantic_generator import field_schema_to_pydantic_class
from typing import List, Dict, Optional
from pydantic import model_validator

logger = logging.getLogger(__name__)


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
    labels: Optional[List[str]]

    _default_output_field_description: str = "The classification label"

    @model_validator(mode="after")
    def validate_response_model(self):

        if self.response_model:
            raise NotImplementedError("Classification skill does not support custom response model yet.")

        if self.field_schema:
            # in case field_schema is already provided, we don't need to parse output template and validate labels
            # just ensure that schema contains exactly one field and enum is presented
            if len(self.field_schema) != 1:
                raise ValueError(
                    f"Classification skill only supports one output field, got {self.field_schema}"
                )
            schema = next(iter(self.field_schema.values()))
            if schema["type"] != "string":
                raise ValueError(
                    f"Classification skill only supports string output field, got {schema['type']}"
                )
            if "enum" not in schema:
                raise ValueError(
                    f"Classification skill output field must have enum, got {schema}"
                )
            if not schema["enum"]:
                raise ValueError(
                    f"Classification skill output field enum must not be empty, got {schema}"
                )

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

        self.response_model = field_schema_to_pydantic_class(self.field_schema, self.name, self.description)
        return self
