from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field
from typing import List, Type


class Entity(BaseModel):
    quote_string: str = Field(
        ...,
        description="The text of the entity extracted from the input document.",
    ),
    label: str = Field(
        ...,
        description="The label assigned to the entity.",
    )


class ExtractedEntities(BaseModel):
    entities: List[Entity] = Field(
        ...,
        description="List of entities extracted from the input document.",
    )


class EntityExtraction(TransformSkill):

    name: str = "entity_extraction"
    input_template: str = 'Extract entities from the input text.\n\nInput:\n"""\n{text}\n"""'
    response_model: Type[BaseModel] = ExtractedEntities
