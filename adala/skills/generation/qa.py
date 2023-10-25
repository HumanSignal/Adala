from .base import TextGenerationSkill


class QuestionAnsweringSkill(TextGenerationSkill):
    """
    Skill specialized for answering questions based on the provided input.

    Inherits from the TextGenerationSkill and focuses on generating answers to the questions 
    posed in the input. The class customizes the instructions, input, and output templates 
    specifically for question-answering tasks.

    Attributes:
        instructions (str): Instruction to guide the LLM in answering the question.
        input_template (str): Format in which the question is presented to the LLM.
        output_template (str): Expected format of the LLM's answer.
        prediction_field (str): Field name for the generated answer.
    """
    
    instructions: str = 'Answer the question.'
    input_template: str = "Question: {{{{{input}}}}}"
    output_template: str = "Answer: {{gen 'answer'}}"
    prediction_field: str = 'answer'

