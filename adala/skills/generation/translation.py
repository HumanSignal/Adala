from .base import TextGenerationSkill


class TranslationSkill(TextGenerationSkill):
    """
    Skill specialized for translation texts based on the provided input.

    Inherits from the TextGenerationSkill and focuses on generating concise summaries
    for the input texts. The class customizes the instructions, input, and output templates
    specifically for text summarization tasks.

    Attributes:
        instructions (str): Instruction to guide the LLM in summarizing the text.
        input_template (str): Format in which the full text is presented to the LLM.
        output_template (str): Expected format of the LLM's summary.
        prediction_field (str): Field name for the generated summary.
    """

    instructions: str = 'Translate the text.'
    input_template: str = "Text: {{{{{input}}}}}"
    output_template: str = "Translation: {{gen 'translation'}}"
    prediction_field: str = 'translation'
