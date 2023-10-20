from .base import TextGenerationSkill


class QuestionAnsweringSkill(TextGenerationSkill):
    instructions = 'Answer the question.'
    input_template = "Question: {{question}}"
    output_template = "Answer: {{gen 'answer'}}"
    prediction_field = 'answer'

