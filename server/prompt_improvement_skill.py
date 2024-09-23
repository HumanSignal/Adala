from pydantic import BaseModel, field_validator
from adala.skills import Skill
from typing import Any, Dict, List, Tuple
from adala.skills.collection.text_generation import TextGenerationSkill


class PromptImprovementSkillResponseModel(BaseModel):
    
    # hidden variable, used for validation
    _input_variables: List[str]
    reasoning: str
    improved_user_prompt: str
    # NOTE: not exposed in LSE yet, so default is always used. Should improve this as well when we expose it.
    # improved_system_prompt: str


    @field_validator("improved_user_prompt", mode="after")
    def validate_used_variables(cls, value: str) -> str:

        start_variable_idx = value.find("{")
        end_variable_idx = value.rfind("}")
        if start_variable_idx == -1 or end_variable_idx == -1 or start_variable_idx >= end_variable_idx:
            raise ValueError("At least one input variable must be used in the prompt")

        try:
            value.format(**{var: "value" for var in cls._input_variables})
        except KeyError as e:
            raise ValueError(f"Invalid variable used in prompt: {e}. Valid variables are: {cls._input_variables}")

        return value
    

def get_prompt_improvement_inputs(student_skill: Skill, input_variables: List[str], student_model: str) -> Dict[str, Any]:
    return {
        "model": student_model,
        "task_name": student_skill.name,
        "task_description": student_skill.description,
        "input_variables": input_variables,
        "current_system_prompt": student_skill.instructions,
        "current_user_prompt": student_skill.input_template,
        "response_json_schema": student_skill.response_model.model_json_schema(),
    }


def get_prompt_improvement_skill(input_variables: List[str]) -> TextGenerationSkill:

    # setting this dynamically - used to validate the improved prompt
    PromptImprovementSkillResponseModel._input_variables = input_variables
    
    prompt_improvement_skill = TextGenerationSkill(
        name="prompt_improvement",
        instructions="Improve the user prompt for the provided LLM model to complete the task using the provided input variables, with the provided user prompt as a starting point. Variables can be accessed in the user prompt using the format {variable_name} (only the variable values are used, not their names). Make sure your prompt produces output that will continue to conform to the provided json schema. Provide your reasoning for the changes you made to the prompt.",
        input_template='''
        Model: {model}
        Task Name: {task_name}
        Task Description: {task_description}
        Input Variables: {input_variables}
        Current System Prompt: {current_system_prompt}
        Current User Prompt: {current_user_prompt}
        Response JSON Schema: {response_json_schema}''',
        response_model=PromptImprovementSkillResponseModel
    )
    
    return prompt_improvement_skill
