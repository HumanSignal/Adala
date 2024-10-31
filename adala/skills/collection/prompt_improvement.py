import json
import logging
from pydantic import BaseModel, field_validator, Field, ConfigDict, model_validator, AfterValidator
from adala.skills import Skill
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from adala.skills import AnalysisSkill
from adala.utils.parse import parse_template
from adala.utils.types import ErrorResponseModel
from adala.skills.collection.label_studio import LabelStudioSkill

logger = logging.getLogger(__name__)


def validate_used_variables(value: str) -> str:
    templates = parse_template(value, include_texts=False)
    if not templates:
        raise ValueError("At least one input variable must be used in the prompt, formatted with curly braces like this: {input_variable}")
    return value


class PromptImprovementSkillResponseModel(BaseModel):

    reasoning: str = Field(
        ..., description="The reasoning for the changes made to the prompt"
    )
    new_prompt_title: str = Field(..., description="The new short title for the prompt")
    new_prompt_content: Annotated[str, AfterValidator(validate_used_variables)]


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
    instructions: str = "Improve current prompt"
    input_template: str = ""  # Used to provide a few shot examples of input-output pairs
    input_prefix: str = ""  # Used to provide additional context for the input
    input_separator: str = "\n"

    response_model = PromptImprovementSkillResponseModel

    @model_validator(mode="after")
    def validate_prompts(self):
        
        def get_json_template(fields):
            json_body = ", ".join([f'"{field}": "{{{field}}}"' for field in fields])
            return "{" + json_body + "}"
        
        if isinstance(self.skill_to_improve, LabelStudioSkill):
            model_json_schema = self.skill_to_improve.field_schema
        else:
            model_json_schema = self.skill_to_improve.response_model.model_json_schema()

        # TODO: can remove this when only LabelStudioSkill is supported
        label_config = getattr(self.skill_to_improve, 'label_config', '<View>Not available</View>')

        input_variables = self.input_variables
        output_variables = list(model_json_schema['properties'].keys())
        input_json_template = get_json_template(input_variables)
        output_json_template = get_json_template(output_variables)
        self.input_template = f'{input_json_template} --> {output_json_template}'
                
        self.input_prefix = f'''
## Current prompt:
```
{self.skill_to_improve.input_template}
```

## Current Labeling Config:
```xml
{label_config}
```

## Input variables:
```
{input_variables}
```

## Model response schema:
```json
{json.dumps(model_json_schema, indent=2)}
```

## Input-Output Examples:

'''
        
        # TODO: deprecated, leave self.output_template for compatibility
        self.output_template = output_json_template
        
        logger.debug(f'Instructions: {self.instructions}\nInput template: {self.input_template}\nInput prefix: {self.input_prefix}')
        return self
