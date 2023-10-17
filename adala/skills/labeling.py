from .base import ProcessingSkill


class TextGenerationSkill(ProcessingSkill):
    prompt_template: str = '''\
    {{instructions}}

    Input: {{text}}
    Output: {{gen 'predictions'}}
    '''


class ClassificationSkill(ProcessingSkill):
    prompt_template: str = '''\
    {{instructions}}
    
    Input: {{text}}
    Output: {{select 'predictions' options=labels logprobs='score'}}
    '''


class ClassificationSkillWithReasoning(ProcessingSkill):
    prompt_template: str = '''\
    {{instructions}}
    Describe your reasoning step-by-step then provide your output.
    
    Input: {{text}}
    Reasoning: {{gen 'rationale'}}
    Output: {{select 'predictions' options=labels logprobs='score'}}
    '''
