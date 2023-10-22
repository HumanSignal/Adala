from ..base import LLMSkill
from typing import List


class ClassificationSkill(LLMSkill):
    instructions: str = 'Label the input text with the following labels: {{{{labels}}}}'
    labels: List[str]
    output_template: str = "Output: {{{{select 'predictions' options=labels logprobs='score'}}}}"
    prediction_field: str = 'predictions'


class ClassificationSkillWithCoT(LLMSkill):
    instructions: str = 'Label the input text with the following labels: {{{{labels}}}}'
    labels: List[str]
    input_template: str = "Input: {{{{{input}}}}}\nThoughts: {{{{gen 'rationale'}}}}\n"
    output_template: str = "Output: {{{{select 'predictions' options=labels logprobs='score'}}}}"
    prediction_field: str = 'predictions'
