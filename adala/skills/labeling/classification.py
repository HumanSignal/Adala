from ..base import LLMSkill
from typing import List


class ClassificationSkill(LLMSkill):
    """
    Skill specialized for classifying text inputs based on a predefined set of labels.

    Args:
        instructions (str): Templated instruction to guide the LLM in classification.
        labels (List[str]): A list of valid labels for the classification task.
        output_template (str): Templated string to format the output from the LLM.
        prediction_field (str): Specifies the field in which predictions will be stored.    
    """
    
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    labels: List[str]
    output_template: str = "Output: {{select 'predictions' options=labels logprobs='score'}}"
    prediction_field: str = 'predictions'


class ClassificationSkillWithCoT(ClassificationSkill):
    """
    Skill specialized for classifying text inputs with the addition of generating a Chain of Thought.

    Args:
        instructions (str): Templated instruction to guide the LLM in classification and to generate a rationale.
        labels (List[str]): A list of valid labels for the classification task.
        input_template (str): Templated string to format the input, which includes a rationale (thoughts).
        output_template (str): Templated string to format the output from the LLM.
        prediction_field (str): Specifies the field in which predictions will be stored.
    """
    
    instructions: str = 'Label the input text with the following labels: {{labels}}. Provide a rationale for your answer.'
    output_template: str = "Thoughts: {{gen 'rationale'}}\nOutput: {{select 'predictions' options=labels logprobs='score'}}"
