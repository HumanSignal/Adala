from .base import Skill
from typing import List


class TextGenerationSkill(Skill):
    instructions: str = 'Generate text based on the provided input.'
    input_template: str = "Input: {{text}}"
    output_template: str = "Output: {{gen 'predictions'}}"
    prediction_field: str = 'predictions'


class LabelingSkill(Skill):
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    labels: List[str]
    input_template: str = "Input: {{text}}"
    output_template: str = "Output: {{select 'predictions' options=labels logprobs='score'}}"
    prediction_field: str = 'predictions'


class LabelingSkillWithCoT(Skill):
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    labels: List[str]
    input_template: str = "Input: {{text}}\nThoughts: {{gen 'rationale'}}\n"
    output_template: str = "Output: {{select 'predictions' options=labels logprobs='score'}}"
    prediction_field: str = 'predictions'
