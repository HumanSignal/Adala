from .base import TextGenerationSkill


class TranslationSkill(TextGenerationSkill):
    name: str = 'translation'
    description: str = 'Translate text from one language to another.'
    instructions: str = 'Identify the language of the given text and translate it to {{target_language}}.'
    input_template: str = "Text: {{{{{input}}}}}"
    # output_template: str = "Input language: {{gen 'detected_language'}}\nTranslation: {{gen 'translation'}}"
    output_template: str = "Translation: {{gen 'translation'}}"
    prediction_field: str = 'translation'
    target_language: str = 'English'
