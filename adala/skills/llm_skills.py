from .base import Skill


class GenerationSkill(Skill):
    prompt_template: str = '''\
    {{instructions}}

    Input: {{input}}
    Output: {{gen 'predictions'}}
    '''


class LabelingSkill(Skill):
    prompt_template: str = '''\
    {{instructions}}
    
    Input: {{input}}
    Output: {{select 'predictions' options=labels logprobs='score'}}
    '''


class LabelingSkillWithReasoning(Skill):
    prompt_template: str = '''\
    {{instructions}}
    Describe your reasoning step-by-step then provide your output.
    
    Input: {{input}}
    Reasoning: {{gen 'rationale'}}
    Output: {{select 'predictions' options=labels logprobs='score'}}
    '''
