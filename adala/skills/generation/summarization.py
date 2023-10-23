from .base import TextGenerationSkill


class SummarizationSkill(TextGenerationSkill):
    instructions: str = 'Summarize the text.'
    input_template: str = "Text: {{{{{input}}}}}"
    output_template: str = "Summary: {{gen 'summary'}}"
    prediction_field: str = 'summary'
