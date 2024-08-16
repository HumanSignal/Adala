import logging
import pandas as pd

from adala.runtimes import Runtime, AsyncRuntime
from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field, model_validator
from typing import List, Type, Optional, Dict

from adala.utils.internal_data import InternalDataFrame
from adala.utils.parse import parse_template
from adala.utils.pydantic_generator import field_schema_to_pydantic_class

logger = logging.getLogger(__name__)


class EntityExtraction(TransformSkill):
    """
    Extract entities from the input text.

    """

    name: str = "entity_extraction"
    input_template: str = 'Extract entities from the input text.\n\nInput:\n"""\n{text}\n"""'
    labels: Optional[List[str]] = None
    output_template: str = 'Extracted entities: {entities}'

    _quote_string_field_name: str = "quote_string"
    _default_quote_string_description: str = "The text of the entity extracted from the input document."
    _labels_field_name: str = "label"
    _default_labels_description: str = "The label assigned to the entity."

    @model_validator(mode="after")
    def validate_response_model(self):
        if self.response_model:
            raise NotImplementedError("EntityExtraction skill does not support custom response model yet.")

        if self.field_schema:
            # in case field_schema is already provided, we don't need to parse output template and validate labels
            # just ensure that schema contains exactly one field and enum is presented
            if len(self.field_schema) != 1:
                raise ValueError(f"EntityExtraction skill only supports one output field, got {self.field_schema}")
            schema = next(iter(self.field_schema.values()))
            if schema["type"] != "array":
                raise ValueError(f"EntityExtraction skill only supports array output field, got {schema['type']}")
            if "items" not in schema:
                raise ValueError(f"EntityExtraction skill output field must have items, got {schema}")
            if "type" not in schema["items"]:
                raise ValueError(f"EntityExtraction skill output field items must have type, got {schema['items']}")
            if schema["items"]["type"] != "object":
                raise ValueError(f"EntityExtraction skill output field items must be object, got {schema['items']['type']}")
            if "properties" not in schema["items"] or not schema["items"]["properties"]:
                raise ValueError(f"EntityExtraction skill output field items must have properties, got {schema['items']}")

            if len(schema["items"]["properties"]) > 2:
                raise ValueError(f"EntityExtraction skill output field items properties must have 1 or 2 fields, got {schema['items']['properties']}")

            # identify quote string field as one field of "type": "string" without "enum"
            self._quote_string_field_name = next(
                (k for k, v in schema["items"]["properties"].items()
                 if v["type"] == "string" and "enum" not in v), None)
            if not self._quote_string_field_name:
                raise ValueError(f"EntityExtraction skill output field items properties must contain one field "
                                 f"of type string without enum (quote string), got {schema['items']['properties']}")

            # check if description provided for quote string field, if not - warning and generate a default one
            if "description" not in schema["items"]["properties"][self._quote_string_field_name]:
                logger.warning(f"EntityExtraction skill output field items properties quote string field must have 'description', "
                               f"generated default description: {self._default_quote_string_description}")
                schema["items"]["properties"][self._quote_string_field_name]["description"] = self._default_quote_string_description

            if len(schema["items"]["properties"]) == 2:
                # identify labels list field as one field of "type": "string" with "enum"
                self._labels_field_name = next(
                    (k for k, v in schema["items"]["properties"].items()
                     if v["type"] == "string" and "enum" in v and v['enum']), None)
                if not self._labels_field_name:
                    raise ValueError(f"EntityExtraction skill output field items properties must have one field"
                                     f" of type string with enum (list of labels), got {schema['items']['properties']}")
                # check if description provided for labels field, if not - warning and generate a default one
                if "description" not in schema["items"]["properties"][self._labels_field_name]:
                    logger.warning(f"EntityExtraction skill output field items properties labels field must have 'description', "
                                   f"generated default description: {self._default_labels_description}")
                    schema["items"]["properties"][self._labels_field_name]["description"] = self._default_labels_description

        else:
            # if field_schema is not provided, parse output_template and validate labels
            output_fields = self.get_output_fields()
            if len(output_fields) != 1:
                raise ValueError(
                    f"Classification skill only supports one output field, got {output_fields}"
                )
            output_field_name = output_fields[0]

            self.field_schema = {
                output_field_name: {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            self._quote_string_field_name: {
                                "type": "string",
                                "description": self._default_quote_string_description
                            }
                        }
                    }
                }
            }
            if self.labels:
                self.field_schema[output_field_name]["items"]["properties"][self._labels_field_name] = {
                    "type": "string",
                    "description": self._default_labels_description,
                    "enum": self.labels
                }

        self.response_model = field_schema_to_pydantic_class(self.field_schema, self.name, self.description)
        return self

    def extract_indices(self, df):
        """
        Give the input dataframe with "text" column and "entities" column of the format
        ```
        [{"quote_string": "entity_1"}, {"quote_string": "entity_2"}, ...]
        ```
         extract the indices of the entities in the input text and put indices in the "entities" column:
         ```
         [{"quote_string": "entity_1", "start": 0, "end": 5}, {"quote_string": "entity_2", "start": 10, "end": 15}, ...]
         ```
        """
        for i, row in df.iterrows():
            text = row["text"]
            entities = row["entities"]
            to_remove = []
            for entity in entities:
                # TODO: current naive implementation assumes that the quote_string is unique in the text.
                # this can be as a baseline for now
                # and we can improve this to handle entities ambiguity (for example, requesting "prefix" in response model)
                # as well as fuzzy pattern matching
                start_idx = text.lower().find(entity["quote_string"].lower())
                if start_idx == -1:
                    # we need to remove the entity if it is not found in the text
                    to_remove.append(entity)
                else:
                    entity["start"] = start_idx
                    entity["end"] = start_idx + len(entity["quote_string"])
            for entity in to_remove:
                entities.remove(entity)
        return df

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        output = super().apply(input, runtime)
        output = self.extract_indices(pd.concat([input, output], axis=1))
        return output

    async def aapply(
        self,
        input: InternalDataFrame,
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:
        output = await super().aapply(input, runtime)
        output = self.extract_indices(pd.concat([input, output], axis=1))
        return output
