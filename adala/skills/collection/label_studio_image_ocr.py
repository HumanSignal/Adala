import re
import logging
import pandas as pd
from typing import List, Optional, Type
from functools import cached_property
from copy import deepcopy
from collections import defaultdict
import aiohttp
import base64
import asyncio
import io
from PIL import Image
from urllib.parse import urlparse
import uuid
from adala.skills._base import TransformSkill
from adala.runtimes import AsyncLiteLLMVisionRuntime
from adala.runtimes._litellm import MessageChunkType
from pydantic import BaseModel, Field, model_validator, computed_field
from difflib import SequenceMatcher


from adala.runtimes import Runtime, AsyncRuntime
from adala.utils.internal_data import InternalDataFrame

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk._extensions.label_studio_tools.core.utils.json_schema import (
    json_schema_to_pydantic,
)


logger = logging.getLogger(__name__)


def extract_variable_name(input_string):
    """Extract variable name in which would be specified as $<variable-name>"""
    pattern = r"\$([a-zA-Z0-9_]+)"
    matches = re.findall(pattern, input_string)
    return matches


class LabelStudioSkillImageOCR(TransformSkill):

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
            
    @classmethod
    async def process_images_with_ocr(cls, images: list) -> list:
        """
        Process a list of images with OCR by calling the OCR service.
        
        Args:
            images: List of image data (URLs or base64 strings)
            
        Returns:
            List of OCR results for each image
        """
        
        async def process_single_image(image_data):
            # Check if the image is a URL
            is_url = False
            try:
                parsed = urlparse(image_data)
                is_url = all([parsed.scheme, parsed.netloc])
            except:
                is_url = False
                
            if not is_url:
                logger.warning(f"Image data is not a URL. OCR service requires URLs or base64 data.")
                return None
            
            # Download the image and convert to base64
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(image_data) as response:
                        if response.status == 200:
                            image_bytes = await response.read()
                            # Get image dimensions
                            image = Image.open(io.BytesIO(image_bytes))
                            width, height = image.size
                            # Convert to base64
                            base64_data = base64.b64encode(image_bytes).decode('utf-8')
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to download image: {response.status}, {error_text}")
                            return None
                except Exception as e:
                    logger.error(f"Error downloading image: {str(e)}")
                    return None
            
            # Call the OCR service with base64 data
            ocr_url = "https://llm-ocr-server.appx.humansignal.com/ocr/base64"
            
            # Prepare form data - this is the key change
            form_data = aiohttp.FormData()
            form_data.add_field('image_data', base64_data)
            form_data.add_field('confidence_threshold', str(0.3))
            form_data.add_field('languages', 'en,ch_sim')
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(ocr_url, data=form_data) as response:
                        if response.status == 200:
                            json_response = await response.json()
                            return {
                                "ocr_data": json_response,
                                "image_width": width,
                                "image_height": height
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"OCR service returned error: {response.status}, {error_text}")
                            return None
                except Exception as e:
                    logger.error(f"Error calling OCR service: {str(e)}")
                    return None
        
        # Process all images concurrently
        tasks = [process_single_image(image) for image in images]
        results = await asyncio.gather(*tasks)
        
        # normalize EasyOCR results to RectangleLabels bounding boxes format of Label Studio
        normalized_results = []
        for result in results:
            if not result:
                continue
                
            # Extract OCR response data
            bboxes = result.get('ocr_data', {}).get('bboxes', [])
            texts = result.get('ocr_data', {}).get('texts', [])
            scores = result.get('ocr_data', {}).get('scores', [])
            original_width = result.get('image_width', 1000)
            original_height = result.get('image_height', 1000)
            
            # Convert to Label Studio format
            label_studio_results = []
            
            for i, (bbox, text, score) in enumerate(zip(bboxes, texts, scores)):
                # EasyOCR bboxes format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # We need to convert to Label Studio format with x,y as top-left corner
                # and width, height as percentages
                
                # Calculate top-left corner (minimum x and y)
                x_values = [point[0] for point in bbox]
                y_values = [point[1] for point in bbox]
                
                min_x = min(x_values)
                min_y = min(y_values)
                
                # Calculate width and height
                max_x = max(x_values)
                max_y = max(y_values)
                width = max_x - min_x
                height = max_y - min_y
                
                # Convert to percentages
                x_percent = (min_x / original_width) * 100
                y_percent = (min_y / original_height) * 100
                width_percent = (width / original_width) * 100
                height_percent = (height / original_height) * 100
                
                # generate unique id for the annotation
                id_gen = str(uuid.uuid4())[:8]
                # Create Label Studio format annotation
                annotation = {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rectanglelabels": ['Transcription'],  # TODO: customize
                    "score": score, 
                    "text": text
                }
                
                label_studio_results.append(annotation)
            
            # Replace the OCR result with the Label Studio formatted result
            normalized_results.append(label_studio_results)
            
        return normalized_results
    
    def _calculate_similarity(self, text: str, reference_texts: List[str]) -> float:
        """
        Calculate similarity between a text and a list of reference texts

        Args:
            text: The text to compare
            reference_texts: List of reference texts

        Returns:
            Similarity score between 0 and 1
        """
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        reference_texts = [ref_text.lower() for ref_text in reference_texts]
        
        # Use SequenceMatcher to calculate similarity
        from difflib import SequenceMatcher
        
        best_score = 0
        if reference_texts:
            for ref_text in reference_texts:
                # Calculate similarity ratio using SequenceMatcher
                similarity = SequenceMatcher(None, text, ref_text).ratio()
                print(f"Similarity between {text} and {ref_text}: {similarity}")
                
                # Update best score if this one is higher
                if similarity > best_score:
                    best_score = similarity
        
        return best_score
    
    def _filter_ocr_results(self, ocr_results: list, reference_texts: List[str]) -> list:
        """
        Filter OCR results based on similarity to output texts
        
        Args:
            ocr_results: List of OCR results
            reference_texts: List of reference texts
                
        Returns:
            List of filtered OCR results
        """
        filtered_results = []
        for result in ocr_results:
            # Simple similarity function - can be replaced with more sophisticated methods
            similarity = self._calculate_similarity(result['text'], reference_texts)
            if similarity >= 0.9:
                filtered_results.append(result)
            
        return filtered_results
        

    async def aapply(
        self,
        input: InternalDataFrame,
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:

        with json_schema_to_pydantic(self.field_schema) as ResponseModel:
            # special handling to flag image inputs if they exist
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
            output['label'] = await self.process_images_with_ocr(input['image'].tolist())
            
            # Convert OCR results to a format that can be used for similarity matching
            # Process each row individually
            # Process each row to filter OCR results based on similarity to reference texts
            filtered_labels = []
            for i, row in output.iterrows():
                # Filter OCR results based on similarity to reference texts
                filtered_ocr = self._filter_ocr_results(row['label'], row['output'])
                filtered_labels.append(filtered_ocr)
            output['label'] = filtered_labels
            return output


if __name__ == "__main__":
    images = [
      "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0000.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0001.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0002.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0003.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0004.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0005.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0006.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0007.png",
    #   "https://htx-pub.s3.amazonaws.com/demo/ocr/pdf/output_0008.png"
    ]
    results = asyncio.run(LabelStudioSkillImageOCR.process_images_with_ocr(images))
    print(results)
