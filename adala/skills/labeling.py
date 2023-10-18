from .base import Skill
from typing import List


class TextGenerationSkill(Skill):
    prompt_template: str = '''\
    {{instructions}}

    Input: {{text}}
    Output: {{gen 'predictions'}}
    '''


class LabelingSkill(Skill):
    labels: List[str]
    instructions: str = 'Label the input text with the following labels: {{labels}}'
    prompt_template: str = '''\
    {{>instructions}}
    
    Input: {{text}}
    Output: {{select 'predictions' options=labels logprobs='score'}}
    '''


class LabelingSkillWithReasoning(LabelingSkill):
    prompt_template: str = '''\
    {{instructions}}
    Describe your reasoning step-by-step then provide your output.
    
    Input: {{text}}
    Reasoning: {{gen 'rationale'}}
    Output: {{select 'predictions' options=labels logprobs='score'}}
    '''
