from .base import TextGenerationSkill


class QuestionAnsweringSkill(TextGenerationSkill):
    instructions: str = 'Answer the question.'
    input_template: str = "Question: {{{{{input}}}}}"
    output_template: str = "Answer: {{{{{gen 'answer'}}}}}"
    prediction_field: str = 'answer'

