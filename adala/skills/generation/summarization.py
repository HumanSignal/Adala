from .base import TextGenerationSkill


class SummarizationSkill(TextGenerationSkill):
    """
    Skill specialized for summarizing lengthy texts based on the provided input.

    Inherits from the TextGenerationSkill and focuses on generating concise summaries 
    for the input texts. The class customizes the instructions, input, and output templates 
    specifically for text summarization tasks.

    Attributes:
        instructions (str): Instruction to guide the LLM in summarizing the text.
        input_template (str): Format in which the full text is presented to the LLM.
        output_template (str): Expected format of the LLM's summary.
        prediction_field (str): Field name for the generated summary.
    """
    
    instructions: str = 'Summarize the text.'
    input_template: str = "Text: {{{{{input}}}}}"
    output_template: str = "Summary: {{gen 'summary'}}"
    prediction_field: str = 'summary'
