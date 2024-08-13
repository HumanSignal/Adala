import pandas as pd

from adala.runtimes import Runtime, AsyncRuntime
from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field, model_validator
from typing import List, Type, Optional, Dict

from adala.utils.internal_data import InternalDataFrame


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
