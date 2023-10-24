from ..base import LLMSkill
from typing import List


class SequenceLabelingSkill(LLMSkill):
    """
    Skill specialized for sequence labeling on text inputs based on a predefined set of labels.

    This involves tasks like named entity recognition where each word/token in the sequence 
    might be assigned a label.

    Args:
        instructions (str): Templated instruction to guide the LLM in sequence labeling.
        labels (List[str]): A list of valid labels for the sequence labeling task.
        input_template (str): Templated string to format the input for the LLM.
        output_template (str): Templated string to format the output from the LLM.
        prediction_field (str): Specifies the field in which predictions will be stored.

    Note:
        This class is still a work in progress.
    """
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    labels: List[str]
    input_template: str = "Input: {{{{{input}}}}}"
    output_template: str = "Output: {{select 'predictions' options=labels logprobs='score'}}"
    prediction_field: str = 'predictions'
