import logging
from typing import Dict, Any, Type
from functools import cached_property
from adala.skills._base import TransformSkill
from pydantic import BaseModel, Field, model_validator

from adala.runtimes import Runtime, AsyncRuntime
from adala.utils.internal_data import InternalDataFrame

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk._extensions.label_studio_tools.core.utils.json_schema import json_schema_to_pydantic


logger = logging.getLogger(__name__)


class LabelStudioSkill(TransformSkill):

    name: str = "label_studio"
    input_template: str = "Annotate the input data according to the provided schema."
    # TODO: remove output_template, fix calling @model_validator(mode='after') in the base class
    output_template: str = "Output: {field_name}"
    response_model: Type[BaseModel] = BaseModel  # why validate_response_model is called in the base class?
    # ------------------------------
    label_config: str = "<View></View>"

    # TODO: implement postprocessing like in EntityExtractionSkill or to verify Taxonomy

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
            return await runtime.batch_to_batch(
                input,
                input_template=self.input_template,
                output_template="",
                instructions_template=self.instructions,
                response_model=ResponseModel,
            )
