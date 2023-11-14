from adala.skills._base import TransformSkill
from typing import List, Dict
from pydantic import model_validator


class ClassificationSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    """
    name: str = 'classification'
    instructions: str = 'Label the input text with the following labels: {labels}'
    input_template: str = 'Input: {text}'
    output_template: str = 'Output: {label}'
    labels: Dict[str, List[str]]

    @model_validator(mode='after')
    def validate_labels(self):
        output_fields = self.get_output_fields()
        self.field_schema = {}
        for labels_field, labels in self.labels.items():
            if labels_field not in output_fields:
                raise ValueError(f'Labels class "{labels_field}" not in output fields {output_fields}')

            self.field_schema[labels_field] = {
                'type': 'array',
                'items': {
                    'type': 'string',
                    'enum': labels
                }
            }
