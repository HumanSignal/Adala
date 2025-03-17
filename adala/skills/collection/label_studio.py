import re
import logging
import pandas as pd
from typing import List, Optional, Type
from functools import cached_property
from copy import deepcopy
from collections import defaultdict
from adala.skills._base import TransformSkill
from adala.runtimes import AsyncLiteLLMVisionRuntime
from adala.runtimes._litellm import MessageChunkType
from pydantic import BaseModel, Field, model_validator, computed_field

from adala.runtimes import Runtime, AsyncRuntime
from adala.utils.internal_data import InternalDataFrame

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk._extensions.label_studio_tools.core.utils.json_schema import (
    json_schema_to_pydantic,
)

from .entity_extraction import extract_indices, validate_output_format_for_ner_tag

logger = logging.getLogger(__name__)


def extract_variable_name(input_string):
    """Extract variable name in which would be specified as $<variable-name>"""
    pattern = r"\$([a-zA-Z0-9_]+)"
    matches = re.findall(pattern, input_string)
    return matches


class LabelStudioSkill(TransformSkill):

    name: str = "label_studio"
    input_template: str = "Annotate the input data according to the provided schema."
    # TODO: remove output_template, fix calling @model_validator(mode='after') in the base class
    output_template: str = "Output: {field_name}"
    response_model: Type[BaseModel] = (
        BaseModel  # why validate_response_model is called in the base class?
    )
    # ------------------------------
    label_config: str = "<View></View>"
    allowed_control_tags: Optional[list[str]] = None
    allowed_object_tags: Optional[list[str]] = None

    # TODO: implement postprocessing to verify Taxonomy

    @cached_property
    def label_interface(self) -> LabelInterface:
        return LabelInterface(self.label_config)

    @cached_property
    def ner_tags(self) -> List[ControlTag]:
        # check if the input config has NER tag (<Labels> + <Text>), and return its `from_name` and `to_name`
        control_tag_names = self.allowed_control_tags or list(
            self.label_interface._controls.keys()
        )
        tags = []
        for tag_name in control_tag_names:
            tag = self.label_interface.get_control(tag_name)
            if tag.tag.lower() in {"labels", "hypertextlabels"}:
                if self.allowed_object_tags:
                    if all(
                        object_tag.tag in self.allowed_object_tags
                        for object_tag in tag.objects
                    ):
                        tags.append(tag)
                else:
                    tags.append(tag)
        return tags

    @cached_property
    def image_tags(self) -> List[ObjectTag]:
        # check if any image tags are used as input variables
        object_tag_names = self.allowed_object_tags or list(
            self.label_interface._objects.keys()
        )
        tags = []
        for tag_name in object_tag_names:
            tag = self.label_interface.get_object(tag_name)
            if tag.tag.lower() == "image":
                tags.append(tag)
        return tags

    def __getstate__(self):
        """Exclude cached properties when pickling - otherwise the 'Agent' can not be serialized in celery"""
        state = deepcopy(super().__getstate__())
        # Remove cached_property values
        for key in ["label_interface", "ner_tags", "image_tags"]:
            state["__dict__"].pop(key, None)
        return state

    @model_validator(mode="after")
    def validate_response_model(self):

        logger.debug(f"Read labeling config {self.label_config}")

        if self.allowed_control_tags or self.allowed_object_tags:
            if self.allowed_control_tags:
                control_tags = {
                    tag: self.label_interface._controls[tag]
                    for tag in self.allowed_control_tags
                }
            else:
                control_tags = self.label_interface._controls
            if self.allowed_object_tags:
                object_tags = {
                    tag: self.label_interface._objects[tag]
                    for tag in self.allowed_object_tags
                }
            else:
                object_tags = self.label_interface._objects
            interface = LabelInterface.create_instance(
                tags={**control_tags, **object_tags}
            )
            logger.debug(
                f"Filtered labeling config based on allowed tags {self.allowed_control_tags=} and {self.allowed_object_tags=} to {interface.config}"
            )
        else:
            interface = self.label_interface

        # NOTE: filtered label config is used for the response model, but full label config is used for the prompt, so that the model has as much context as possible.
        self.field_schema = interface.to_json_schema()
        logger.debug(f"Converted labeling config to json schema: {self.field_schema}")

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
            # special handling to flag image inputs if they exist
            if isinstance(runtime, AsyncLiteLLMVisionRuntime):
                input_field_types = defaultdict(lambda: MessageChunkType.TEXT)
                for tag in self.image_tags:
                    # these are the project variable names, NOT the label config tag names. TODO: pass this info from LSE to avoid recomputing it here.
                    variables = extract_variable_name(tag.value)
                    if len(variables) != 1:
                        logger.warning(
                            f"Image tag {tag.name} has multiple variables: {variables}. Cannot mark these variables as image inputs."
                        )
                        continue
                    input_field_types[variables[0]] = (
                        MessageChunkType.IMAGE_URLS
                        if tag.attr.get("valueList")
                        else MessageChunkType.IMAGE_URL
                    )

                logger.debug(
                    f"Using VisionRuntime with input field types: {input_field_types}"
                )
                output = await runtime.batch_to_batch(
                    input,
                    input_template=self.input_template,
                    output_template="",
                    instructions_template=self.instructions,
                    response_model=ResponseModel,
                    input_field_types=input_field_types,
                )
            else:
                output = await runtime.batch_to_batch(
                    input,
                    input_template=self.input_template,
                    output_template="",
                    instructions_template=self.instructions,
                    response_model=ResponseModel,
                )
            for ner_tag in self.ner_tags:
                input_field_name = ner_tag.objects[0].value.lstrip("$")
                output_field_name = ner_tag.name
                quote_string_field_name = "text"
                df = pd.concat([input, output], axis=1)
                output = validate_output_format_for_ner_tag(
                    df, input_field_name, output_field_name
                )
                output = extract_indices(
                    output, input_field_name, output_field_name, quote_string_field_name
                )
            return output
