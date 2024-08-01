from adala.skills._base import Skill, TransformSkill
from typing import List, Dict
from pydantic import model_validator


class ClassificationSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    """

    name: str = "classification"
    instructions: str = "Classify input text."
    input_template: str = 'Input:\n"""\n{text}\n"""'
    output_template: str = "Classification result: {label}"
    labels: Dict[str, List[str]]

    @model_validator(mode="after")
    def validate_labels(self):
        output_fields = self.get_output_fields()
        if len(output_fields) != 1:
            raise ValueError(
                f"Classification skill only supports one output field, got {output_fields}"
            )
        self.field_schema = {}
        for labels_field, labels in self.labels.items():
            if labels_field not in output_fields:
                raise ValueError(
                    f'Labels class "{labels_field}" not in output fields {output_fields}'
                )

            self.field_schema[labels_field] = {
                "type": "string",
                "description": "The classification label.",
                "enum": labels,
            }
