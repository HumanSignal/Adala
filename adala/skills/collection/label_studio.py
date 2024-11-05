import logging
import pandas as pd
from typing import Optional, Type
from functools import cached_property
from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field, model_validator

from adala.runtimes import Runtime, AsyncRuntime
from adala.utils.internal_data import InternalDataFrame

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.control_tags import ControlTag
from label_studio_sdk._extensions.label_studio_tools.core.utils.json_schema import json_schema_to_pydantic

from .entity_extraction import extract_indices, validate_output_format_for_ner_tag

logger = logging.getLogger(__name__)


class LabelStudioSkill(TransformSkill):

    name: str = "label_studio"
    input_template: str = "Annotate the input data according to the provided schema."
    # TODO: remove output_template, fix calling @model_validator(mode='after') in the base class
    output_template: str = "Output: {field_name}"
    response_model: Type[BaseModel] = BaseModel  # why validate_response_model is called in the base class?
    # ------------------------------
    label_config: str = "<View></View>"

    # TODO: implement postprocessing to verify Taxonomy

    def has_ner_tag(self) -> Optional[ControlTag]:
        # check if the input config has NER tag (<Labels> + <Text>), and return its `from_name` and `to_name`
        interface = LabelInterface(self.label_config)
        for tag in interface.controls:
            if tag.tag == 'Labels':
                return tag
            
    @model_validator(mode='after')
    def validate_response_model(self):

        interface = LabelInterface(self.label_config)
        logger.debug(f'Read labeling config {self.label_config}')

        self.field_schema = interface.to_json_schema()
        logger.debug(f'Converted labeling config to json schema: {self.field_schema}')

        return self
    
    def _create_response_model_from_field_schema(self):
        pass

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        
        with json_schema_to_pydantic(self.field_schema) as ResponseModel:
            return runtime.batch_to_batch(
                input,
                input_template=self.input_template,
                output_template="",
                instructions_template=self.instructions,
                response_model=ResponseModel,
            )

    async def aapply(
        self,
        input: InternalDataFrame,
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:

        with json_schema_to_pydantic(self.field_schema) as ResponseModel:
            output = await runtime.batch_to_batch(
                input,
                input_template=self.input_template,
                output_template="",
                instructions_template=self.instructions,
                response_model=ResponseModel,
            )
            #FIXME: this is only done for the first NER tag; it should be done for each NER tag
            ner_tag = self.has_ner_tag()
            if ner_tag:
                input_field_name = ner_tag.objects[0].value.lstrip('$')
                output_field_name = ner_tag.name
                quote_string_field_name = 'text'
                df = pd.concat([input, output], axis=1)
                output = validate_output_format_for_ner_tag(df, input_field_name, output_field_name)
                output = extract_indices(output, input_field_name, output_field_name, quote_string_field_name)
            return output
