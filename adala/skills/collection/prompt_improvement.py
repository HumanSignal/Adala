import json
import logging
from pydantic import BaseModel, field_validator, Field, ConfigDict, model_validator
from adala.skills import Skill
from typing import Any, Dict, List, Optional, Union
from adala.skills import AnalysisSkill
from adala.utils.parse import parse_template
from adala.utils.types import ErrorResponseModel
from adala.skills.collection.label_studio import LabelStudioSkill

logger = logging.getLogger(__name__)


class PromptImprovementSkillResponseModel(BaseModel):

    reasoning: str = Field(
        ..., description="The reasoning for the changes made to the prompt"
    )
    new_prompt_title: str = Field(..., description="The new short title for the prompt")
    new_prompt_content: str = Field(..., description="The new content for the prompt")

    # model_config = ConfigDict(
    #     # omit other fields
    #     extra="ignore",
    #     # guard against name collisions with other fields
    #     populate_by_name=False,
    # )

    # @field_validator("new_prompt_content", mode="after")
    # def validate_used_variables(cls, value: str) -> str:

    #     templates = parse_template(value, include_texts=False)
    #     if not templates:
    #         raise ValueError("At least one input variable must be used in the prompt")

    #     input_vars_used = [t["text"] for t in templates]
    #     if extra_vars_used := set(input_vars_used) - set(cls._input_variables):
    #         raise ValueError(
    #             f"Invalid variable used in prompt: {extra_vars_used}. Valid variables are: {cls._input_variables}"
    #         )

    #     return value


class ImprovedPromptResponse(BaseModel):

    output: Union[PromptImprovementSkillResponseModel, ErrorResponseModel]

    prompt_tokens: int = Field(alias="_prompt_tokens", default=None)
    completion_tokens: int = Field(alias="_completion_tokens", default=None)

    # these can fail to calculate
    prompt_cost_usd: Optional[float] = Field(alias="_prompt_cost_usd", default=None)
    completion_cost_usd: Optional[float] = Field(
        alias="_completion_cost_usd", default=None
    )
    total_cost_usd: Optional[float] = Field(alias="_total_cost_usd", default=None)


class PromptImprovementSkill(AnalysisSkill):

    skill_to_improve: Skill
    # TODO: include model provider to specialize the prompt format for a specific model provider
    # model_provider: str
    input_variables: List[str]

    name: str = "prompt_improvement"
    instructions: str = ""  # Automatically generated
    input_template: str = ""  # Not used
    input_prefix: str = (
        "Here are a few prediction results after applying the current prompt for your analysis.\n\n"
    )
    input_separator: str = "\n\n"

    response_model = PromptImprovementSkillResponseModel

    @model_validator(mode="after")
    def validate_prompts(self):
        
        input_variables = "\n".join(self.input_variables)
        if isinstance(self.skill_to_improve, LabelStudioSkill):
            model_json_schema = self.skill_to_improve.field_schema
        else:
            model_json_schema = self.skill_to_improve.response_model.model_json_schema()
        # rewrite the instructions with the actual values
        self.instructions = f"""\
You are a prompt engineer tasked with generating or enhancing a prompt for a Language Learning Model (LLM). Your goal is to create an effective prompt based on the given context, input data and requirements.

First, carefully review the following context information:

# Given context

## Task name
{self.skill_to_improve.name}

## Task description
{self.skill_to_improve.description}

## Input variables to use
{input_variables}

## Target response schema
```json
{json.dumps(model_json_schema, indent=2)}
```
Now, examine the current prompt (if provided):

# Current prompt
{self.skill_to_improve.input_template}

If a current prompt is provided, analyze it for potential improvements or errors. Consider how well it addresses the task description, input data and if it effectively utilizes all provided input variables.

Before creating the new prompt, provide a detailed reasoning for your choices. Using only bullet points, list the changes to be made and concisely explain why each change is necessary. Include:
1. How you plan to address the context and task description
2. Specific errors or improvements you've identified in the previous prompt (if applicable)
3. How you intend to tailor the new prompt to better suit the target model provider
4. Your strategy for designing the prompt to generate responses matching the provided schema

Next, generate a new short prompt title that accurately reflects the task and purpose of the prompt.

Finally, create the new prompt content. Ensure that you:
1. Incorporate all provided input variables, formatted with "{{" and "}}" brackets
2. Address the specific task description provided in the context
3. Consider the target model provider's capabilities and limitations
4. Maintain or improve upon any relevant information from the current prompt (if provided)
5. Structure the prompt to elicit a response that matches the provided response schema

Present your output in JSON format including the following fields:
- reasoning
- new_prompt_title
- new_prompt_content


# Example of the expected input and output:

Input context:

## Target model provider
OpenAI

## Task description
Generate a summary of the input text.

## Allowed input variables
text
document_metadata

## Target response schema
```json
{{
    "summary": {{
        "type": "string"
    }},
    "categories": {{
        "type": "string",
        "enum": ["news", "science", "politics", "sports", "entertainment"]
    }}
}}
```

Check the following example to see how the model should respond:

Current prompt:
```
Generate a summary of the input text: "{{text}}".
```

# Current prompt output

Generate a summary of the input text: "The quick brown fox jumps over the lazy dog." --> {{"summary": "The quick brown fox jumps over the lazy dog.", "categories": "news"}}

Generate a summary of the input text: "When was the Battle of Hastings?" --> {{"summary": "The Battle of Hastings was a decisive Norman victory in 1066, marking the end of Anglo-Saxon rule in England.", "categories": "history"}}

Generate a summary of the input text: "What is the capital of France?" --> {{   "summary": "The capital of France is Paris.", "categories": "geography"}}


Your output:
```json
{{
    "reasoning": "Changes needed:
• Specify format and style of summary: The current prompt is vague, leading to inconsistent outputs.
• Add categories instructions: The prompt doesn't mention categorization, resulting in missing or incorrect categories.
• Use all input variables: The 'document_metadata' variable is not utilized in the current prompt.
• Align with response schema: Ensure only allowed categories are used (e.g., remove 'history').
• Improve clarity: Guide the model to summarize the text rather than answering questions.
These changes will ensure higher quality, more consistent responses that meet the specified requirements.",
    "new_prompt_title": "Including categories instructions in the summary",
    "new_prompt_content": "Generate a detailed summary of the input text:\n'''{{text}}'''.\nUse the document metadata to guide the model to produce categories.\n#Metadata:\n'''{{document_metadata}}'''.\nEnsure high quality output by asking the model to produce a detailed summary and to categorize the document."
}}
```

Ensure that your refined prompt is clear, concise, and effectively guides the LLM to produce high quality responses.

"""

        # Create the output template for JSON output based on the response model fields
        fields = self.skill_to_improve.response_model.model_fields
        field_template = ", ".join([f'"{field}": "{{{field}}}"' for field in fields])
        self.output_template = "{{" + field_template + "}}"
        self.input_template = (
            f"{self.skill_to_improve.input_template} --> {self.output_template}"
        )
        return self
