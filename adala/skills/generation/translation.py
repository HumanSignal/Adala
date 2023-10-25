from .base import TextGenerationSkill


class TranslationSkill(TextGenerationSkill):
    """
    Skill specialized for translating text from one language to another.

    Inherits from the TextGenerationSkill and focuses on translating the input text to the
    specified target language. The class customizes the instructions, input, and output templates
    specifically for translation tasks.

    Attributes:
        instructions (str): Instruction to guide the LLM in translating the text.
        input_template (str): Format in which the full text is presented to the LLM.
        output_template (str): Expected format of the LLM's translation.
        prediction_field (str): Field name for the generated translation.
        target_language (str): Language to which the input text is translated.
    """

    name: str = 'translation'
    description: str = 'Translate text from one language to another.'
    instructions: str = 'Identify the language of the given text and translate it to {{target_language}}.'
    input_template: str = "Text: {{{{{input}}}}}"
    # output_template: str = "Input language: {{gen 'detected_language'}}\nTranslation: {{gen 'translation'}}"
    output_template: str = "Translation: {{gen 'translation'}}"
    prediction_field: str = 'translation'
    target_language: str = 'English'
