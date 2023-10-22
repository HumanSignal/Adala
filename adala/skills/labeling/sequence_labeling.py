from ..base import LLMSkill
from typing import List


class SequenceLabelingSkill(LLMSkill):
    # TODO: Work in progress...
    instructions: str = 'Label the input text with the following labels: {{{{labels}}}}'
    labels: List[str]
    input_template: str = "Input: {{{{{input}}}}}"
    output_template: str = "Output: {{{{select 'predictions' options=labels logprobs='score'}}}}"
    prediction_field: str = 'predictions'
