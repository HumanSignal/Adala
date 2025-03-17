import re
import logging
import pandas as pd
from typing import List, Optional, Type, Dict, Tuple
from functools import cached_property
from copy import deepcopy
from collections import defaultdict
import aiohttp
import base64
import asyncio
import io
from thefuzz import fuzz
from PIL import Image
from urllib.parse import urlparse
import uuid
from adala.skills._base import TransformSkill
from adala.runtimes import AsyncLiteLLMVisionRuntime
from adala.runtimes._litellm import MessageChunkType
from pydantic import BaseModel, Field, model_validator, computed_field
from difflib import SequenceMatcher
import numpy as np

from adala.runtimes import Runtime, AsyncRuntime
from adala.utils.internal_data import InternalDataFrame

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk._extensions.label_studio_tools.core.utils.json_schema import (
    json_schema_to_pydantic,
)
from .match_bbox_by_text import find_text_in_image


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
        
        return results
    
    def _get_normalized_bbox(self, bbox: List[List[int]], original_width: int, original_height: int) -> List[float]:
        # Calculate top-left corner (minimum x and y)
        # bbox format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
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
        
        return {
            'x': x_percent,
            'y': y_percent,
            'width': width_percent,
            'height': height_percent
        }
    
    def _convert_ocr_results_to_label_studio_format(self, results: list) -> Tuple[List, List]:
        
        # normalize EasyOCR results to RectangleLabels bounding boxes format of Label Studio
        all_bbox_annotations = []
        all_text_annotations = []
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
            bbox_annotations = []
            text_annotations = []
            
            for i, (bbox, text, score) in enumerate(zip(bboxes, texts, scores)):
                # EasyOCR bboxes format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # We need to convert to Label Studio format with x,y as top-left corner
                # and width, height as percentages
                
                bbox_annotation = self._get_normalized_bbox(bbox, original_width, original_height)
                
                # generate unique id for the annotation
                id_gen = str(uuid.uuid4())[:8]
                # Create Label Studio format annotation
                bbox_annotation['id'] = id_gen
                bbox_annotation['rotation'] = 0
                
                text_annotation = {
                    'text': [text],
                    'id': id_gen
                }
                
                bbox_annotations.append(bbox_annotation)
                text_annotations.append(text_annotation)
            
            # Replace the OCR result with the Label Studio formatted result
            all_bbox_annotations.append(bbox_annotations)
            all_text_annotations.append(text_annotations)
            
        return all_bbox_annotations, all_text_annotations
    
    def _convert_ocr_results_to_label_studio_format_v2(self, results: list) -> Tuple[List, List]:
        """
        Same as _convert_ocr_results_to_label_studio_format, but uses a different approach to filter the OCR results:
        results['ocr_data'] contains a dictionary of reference texts as keys and lists of OCR results as values.
        For each reference text, we create a group of `bbox_annotations` and `text_annotations`. 
        In text_annotations, we place "parentID" as the id of the bbox_annotation that it belongs to (pick the first text_annotation id as parentID)
        """
        all_bbox_annotations = []
        all_text_annotations = []
        
        for result in results:
            bbox_annotations = []
            text_annotations = []
            
            original_width = result['image_width']
            original_height = result['image_height']
            
            # Process each reference text and its associated OCR results
            for reference_text, ocr_matches in result['ocr_data'].items():
                # Create a group for this reference text
                group_id = None
                
                # Process each OCR match for this reference text
                for bbox, text, score in zip(ocr_matches['bboxes'], ocr_matches['texts'], ocr_matches['scores']):
                    
                    bbox_annotation = self._get_normalized_bbox(bbox, original_width, original_height)
                    
                    # Generate unique id for the annotation
                    id_gen = str(uuid.uuid4())[:8]
                    if group_id is None:
                        group_id = id_gen
                    
                    # Create bbox annotation
                    bbox_annotation['rotation'] = 0
                    bbox_annotation['id'] = id_gen
                    bbox_annotation['score'] = score
                    # Create text annotation
                    text_annotation = {
                        'text': [text],
                        'id': id_gen,
                    }
                    if group_id != id_gen:
                        text_annotation['parent_id'] = group_id
                        bbox_annotation['parent_id'] = group_id
                    
                    bbox_annotations.append(bbox_annotation)
                    text_annotations.append(text_annotation)
                
            # Add annotations for this result to the overall lists
            all_bbox_annotations.append(bbox_annotations)
            all_text_annotations.append(text_annotations)
            
        return all_bbox_annotations, all_text_annotations
                
        
    @classmethod
    def _calculate_similarity(cls, text: str, reference_texts: List[str]) -> Tuple[float, str]:
        """
        Calculate similarity between a text and substrings within reference texts.

        Args:
            text: The text to compare
            reference_texts: List of reference texts

        Returns:
            Similarity score between 0 and 1 and the best matching text
        """
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        text_len = len(text)
        
        best_score = 0
        best_match = None
        
        if reference_texts:
            for ref_text in reference_texts:
                ref_text_lower = ref_text.lower()
                best_window_score = fuzz.partial_ratio(text, ref_text_lower)
                if best_window_score > best_score:
                    best_score = best_window_score
                    best_match = ref_text
        best_score = float(best_score) / 100
        print(f"Best substring similarity between '{text}' and '{best_match}': {best_score}")
                    
        return best_score, best_match
    
    def _filter_ocr_results(self, ocr_results: Dict, reference_texts: List[str]) -> list:
        """
        Filter OCR results based on similarity to output texts
        
        Args:
            ocr_results: List of OCR results
            reference_texts: List of reference texts
                
        Returns:
            List of filtered OCR results
        """
        filtered_results = {
            'bboxes': [],
            'texts': [],
            'scores': []
        }
        for bbox, text, score in zip(ocr_results['bboxes'], ocr_results['texts'], ocr_results['scores']):
            # Simple similarity function - can be replaced with more sophisticated methods
            similarity, best_match = self._calculate_similarity(text, reference_texts)
            if similarity >= 0.9:
                filtered_results['bboxes'].append(bbox)
                filtered_results['texts'].append(best_match)
                filtered_results['scores'].append(score)
            
        return filtered_results
    
    def _filter_ocr_results_v2(self, ocr_results: Dict, reference_texts: List[str]) -> Dict[str, List]:
        
        output = {}
        for ref_text in reference_texts:
            ref_text_lower = ref_text.lower()
            output[ref_text] = {
                'bboxes': [],
                'texts': [],
                'scores': []
            }
            for text, score, bbox in zip(ocr_results['texts'], ocr_results['scores'], ocr_results['bboxes']):
                text_lower = text.lower()
                # check if text is a fuzzy substring of ref_text
                similarity = fuzz.partial_ratio(text_lower, ref_text_lower)
                if similarity >= 95:
                    output[ref_text]['bboxes'].append(bbox)
                    output[ref_text]['texts'].append(text)
                    output[ref_text]['scores'].append(score)
            
        # Filter to keep only horizontally aligned bounding boxes
        for ref_text in output:
            if not output[ref_text]['bboxes']:
                continue
                
            # Group bounding boxes by their vertical position (y-coordinate)
            # Using the middle y-coordinate of each box for grouping
            y_groups = {}
            for i, bbox in enumerate(output[ref_text]['bboxes']):
                # Calculate middle y-coordinate of the bounding box
                # bbox format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # Calculate middle y-coordinate of the bounding box
                # bbox format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                y_values = [point[1] for point in bbox]
                mid_y = sum(y_values) / len(y_values)
                
                # Group with tolerance of 10 pixels
                group_key = int(mid_y / 10) * 10
                if group_key not in y_groups:
                    y_groups[group_key] = []
                y_groups[group_key].append(i)
            
            # Find the group with the maximum number of bounding boxes
            max_group_key = max(y_groups.keys(), key=lambda k: len(y_groups[k]), default=None)
            
            if max_group_key is not None:
                # Keep only the bounding boxes in the largest horizontal group
                indices_to_keep = y_groups[max_group_key]
                
                # Create new filtered lists
                filtered_bboxes = [output[ref_text]['bboxes'][i] for i in indices_to_keep]
                filtered_texts = [output[ref_text]['texts'][i] for i in indices_to_keep]
                filtered_scores = [output[ref_text]['scores'][i] for i in indices_to_keep]
                
                # Sort bounding boxes by x-coordinate to maintain reading order
                sorted_indices = sorted(range(len(filtered_bboxes)), 
                                       key=lambda i: min(point[0] for point in filtered_bboxes[i]))
                
                filtered_bboxes = [filtered_bboxes[i] for i in sorted_indices]
                filtered_texts = [filtered_texts[i] for i in sorted_indices]
                filtered_scores = [filtered_scores[i] for i in sorted_indices]
                
                # Create a combined bounding box that encompasses all individual boxes
                if filtered_bboxes:
                    # Find min and max coordinates across all bounding boxes
                    all_x = [point[0] for bbox in filtered_bboxes for point in bbox]
                    all_y = [point[1] for bbox in filtered_bboxes for point in bbox]
                    
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    
                    # Create a new bounding box with the min/max coordinates
                    combined_bbox = [
                        [min_x, min_y],  # top-left
                        [max_x, min_y],  # top-right
                        [max_x, max_y],  # bottom-right
                        [min_x, max_y]   # bottom-left
                    ]
                    
                    # Calculate average score
                    avg_score = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0
                    
                    # Add the combined bounding box to the results
                    filtered_bboxes.insert(0, combined_bbox)
                    filtered_texts.insert(0, ref_text)  # Use the reference text for the combined box
                    filtered_scores.insert(0, avg_score)
                
                # Update the output with filtered results
                output[ref_text]['bboxes'] = filtered_bboxes
                output[ref_text]['texts'] = filtered_texts
                output[ref_text]['scores'] = filtered_scores
        
        return output
    
    
    def _get_labels(self) -> List[str]:
        # TODO: validate labels are coming from <Labels> tag, use control tag name
        # format: {'StartDate': LabelTag(attr={'value': 'StartDate', 'background': 'red'}, tag='Label', value='StartDate', parent_name='columns'), 'EndDate': LabelTag(attr={'value': 'EndDate', 'background': 'green'}, tag='Label', value='EndDate', parent_name='columns'), 'Amount': LabelTag(attr={'value': 'Amount'}, tag='Label', value='Amount', parent_name='columns')}
        return list(self.label_interface.labels)[0]
        

    async def aapply(
        self,
        input: InternalDataFrame,
        runtime: AsyncRuntime,
    ) -> InternalDataFrame:
        
        labels = self._get_labels()
        # validate labels
        from adala.utils.pydantic_generator import field_schema_to_pydantic_class
        LineItem = field_schema_to_pydantic_class(
            class_name="LineItem",
            description="A single line extracted from the document",
            field_schema={label: {"type": "string"} for label in labels}
        )
        
        class ResponseModel(BaseModel):
            lines: List[LineItem]
        
        input_field_types = defaultdict(lambda: MessageChunkType.TEXT)
        image_value_key = None
        for tag in self.image_tags:
            # these are the project variable names, NOT the label config tag names. TODO: pass this info from LSE to avoid recomputing it here.
            variables = extract_variable_name(tag.value)
            if len(variables) != 1:
                logger.warning(
                    f"Image tag {tag.name} has multiple variables: {variables}. Cannot mark these variables as image inputs."
                )
                continue
            image_value_key = variables[0]
            input_field_types[image_value_key] = (
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
        print(f'Output: {output}')
        
        images = input[image_value_key].tolist()
        all_bbox_annotations = []
        all_text_annotations = []
        all_label_annotations = []
        for i, row in output.iterrows():
            extracted_results = row['lines']
            
            ocr_results = find_text_in_image(images[i], extracted_results)
            bbox_annotations = []
            text_annotations = []
            label_annotations = []
            for ocr_result in ocr_results:
                # Add bbox annotation
                
                bbox_id = ocr_result['element']['id']
                parent_id = ocr_result['element'].get('parent_id')
                
                bbox_annotation = ocr_result['element']
                bbox_annotation['score'] = ocr_result['matching_score'] * ocr_result['element']['score']
                
                bbox_annotations.append(ocr_result['element'])
                
                # Add text annotation
                text_annotation = {
                    'text': [ocr_result['reference_text']],
                    'id': bbox_id
                }
                if parent_id:
                    text_annotation['parent_id'] = parent_id
                text_annotations.append(text_annotation)
                
                label = ocr_result.pop('reference_label', None)

                if label:
                    label_annotation = {
                        'labels': [label],
                        'id': bbox_id
                    }
                    if parent_id:
                        label_annotation['parent_id'] = parent_id
                    label_annotations.append(label_annotation)
                    
            all_bbox_annotations.append(bbox_annotations)
            all_text_annotations.append(text_annotations)   
            all_label_annotations.append(label_annotations)
        output['bbox'] = all_bbox_annotations
        output['transcription'] = all_text_annotations
        output['columns'] = all_label_annotations
        return output
            

        # with json_schema_to_pydantic(self.field_schema) as ResponseModel:
        #     # special handling to flag image inputs if they exist
        #     input_field_types = defaultdict(lambda: MessageChunkType.TEXT)
        #     image_value_key = None
        #     for tag in self.image_tags:
        #         # these are the project variable names, NOT the label config tag names. TODO: pass this info from LSE to avoid recomputing it here.
        #         variables = extract_variable_name(tag.value)
        #         if len(variables) != 1:
        #             logger.warning(
        #                 f"Image tag {tag.name} has multiple variables: {variables}. Cannot mark these variables as image inputs."
        #             )
        #             continue
        #         image_value_key = variables[0]
        #         input_field_types[image_value_key] = (
        #             MessageChunkType.IMAGE_URLS
        #             if tag.attr.get("valueList")
        #             else MessageChunkType.IMAGE_URL
        #         )

        #     logger.debug(
        #         f"Using VisionRuntime with input field types: {input_field_types}"
        #     )
        #     output = await runtime.batch_to_batch(
        #         input,
        #         input_template=self.input_template,
        #         output_template="",
        #         instructions_template=self.instructions,
        #         response_model=ResponseModel,
        #         input_field_types=input_field_types,
        #     )
        #     print(f'Output: {output}')
            # print(f'Process images with OCR: {input[image_value_key].tolist()}')
            # # ocr_results = await self.process_images_with_ocr(input[image_value_key].tolist())
            # # filtered_ocr_results = []
            # images = input[image_value_key].tolist()
            # all_bbox_annotations = []
            # all_text_annotations = []
            # for i, row in output.iterrows():
            #     extracted_result = row['output']
                
            #     ocr_results = find_text_in_image(images[i], extracted_result)
            #     bbox_annotations = []
            #     text_annotations = []
            #     for ocr_result in ocr_results:
            #         # Convert OCR results to Label Studio format
            #         parent_id = ocr_result['bbox']['id']

            #         for word in ocr_result['words']:
            #             bbox_annotation = word['bbox']
            #             bbox_annotation['rotation'] = 0
            #             bbox_annotation['parent_id'] = parent_id
            #             bbox_annotation['score'] = word['score']
                                                
            #             text_annotation = {
            #                 'text': [word['text']],
            #                 'id': bbox_annotation['id'],
            #                 'parent_id': parent_id
            #             }
                        
            #             bbox_annotations.append(bbox_annotation)
            #             text_annotations.append(text_annotation)
                        
            #         bbox_annotation = ocr_result['bbox']
            #         bbox_annotation['rotation'] = 0
            #         bbox_annotation['score'] = float(np.sqrt(ocr_result['detection_score'] * ocr_result['matching_score']))
                    
            #         text_annotation = {
            #             'text': [ocr_result['reference_text']],
            #             'id': parent_id
            #         }
                    
            #         bbox_annotations.append(bbox_annotation)
            #         text_annotations.append(text_annotation)
                        
            #     all_bbox_annotations.append(bbox_annotations)
            #     all_text_annotations.append(text_annotations)   
            # output['bbox'] = all_bbox_annotations
            # output['transcription'] = all_text_annotations
            # return output
