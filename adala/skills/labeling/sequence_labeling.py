from ..base import Skill
from typing import List


class SequenceLabelingSkill(Skill):
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    labels: List[str]
    input_template: str = "Input: {{text}}"
    output_template: str = "Output: {{select 'predictions' options=labels logprobs='score'}}"
    prediction_field: str = 'predictions'
