from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field, model_validator
from typing import List, Type, Optional


class EntityExtraction(TransformSkill):

    name: str = "entity_extraction"
    input_template: str = 'Extract entities from the input text.\n\nInput:\n"""\n{text}\n"""'
    labels: Optional[List[str]] = None
    output_template: str = 'Extracted entities: {entities}'

    @model_validator(mode="after")
    def maybe_add_labels(self):
        self.field_schema = {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote_string": {
                            "type": "string",
                            "description": "The text of the entity extracted from the input document."
                        }
                    }
                }
            }
        }
        if self.labels:
            self.field_schema["entities"]["items"]["properties"]["label"] = {
                "type": "string",
                "description": "The label assigned to the entity.",
                "enum": self.labels
            }
