from ..base import LLMSkill
from typing import List


class ClassificationSkill(LLMSkill):
    """
    Skill specialized for classifying text inputs based on a predefined set of labels.

    Attributes:
        name (str): Unique name of the skill.
        instructions (str): Instructs agent what to do with the input data.
        description (str): Description of the skill.
        input_template (str): Template for the input data.
        output_template (str): Template for the output data.
        input_data_field (str): Name of the input data field.
        prediction_field (str): Name of the prediction field to be used for the output data.
        labels (List[str]): A list of valid labels for the classification task.

    Examples:
        >>> import pandas as pd
        >>> skill = ClassificationSkill(labels=['positive', 'negative'])
        >>> skill.apply(pd.DataFrame({'text': ['I love this movie!', 'I hate this movie!']}))
        text                    predictions score
        I love this movie!      positive   0.99
        I hate this movie!      negative   0.99
    """
    
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    labels: List[str]
    output_template: str = "Output: {{select 'predictions' options=labels logprobs='score'}}"
    prediction_field: str = 'predictions'


class ClassificationSkillWithCoT(ClassificationSkill):
    """
    Skill specialized for classifying text inputs with the addition of generating a Chain of Thought.

    Attributes:
        name (str): Unique name of the skill.
        instructions (str): Instructs agent what to do with the input data.
        description (str): Description of the skill.
        input_template (str): Template for the input data.
        output_template (str): Template for the output data.
        input_data_field (str): Name of the input data field.
        prediction_field (str): Name of the prediction field to be used for the output data.
        labels (List[str]): A list of valid labels for the classification task.

    Examples:
        >>> import pandas as pd
        >>> skill = ClassificationSkillWithCoT(labels=['positive', 'negative'])
        >>> skill.apply(pd.DataFrame({'text': ['I love this movie!', 'I hate this movie!']}))
        text                    predictions score  rationale
        I love this movie!      positive   0.99    classified as positive because I love this movie!
        I hate this movie!      negative   0.99    classified as negative because I hate this movie!
    """
    
    instructions: str = 'Label the input text with the following labels: {{labels}}. Provide a rationale for your answer.'
    output_template: str = "Thoughts: {{gen 'rationale'}}\nOutput: {{select 'predictions' options=labels logprobs='score'}}"
