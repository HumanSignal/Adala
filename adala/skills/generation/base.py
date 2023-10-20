from ..base import Skill


class TextGenerationSkill(Skill):
    instructions: str = 'Generate text based on the provided input.'
    input_template: str = "Input: {{text}}"
    output_template: str = "Output: {{gen 'predictions'}}"
    prediction_field: str = 'predictions'
