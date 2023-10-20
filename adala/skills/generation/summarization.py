from .base import TextGenerationSkill


class SummarizationSkill(TextGenerationSkill):
    instructions = 'Summarize the text.'
    input_template = "Text: {{text}}"
    output_template = "Summary: {{gen 'summary'}}"
    prediction_field = 'summary'
