from .._base import TransformSkill


class TextGenerationSkill(TransformSkill):
    """
    Skill specialized for generating text based on the provided input.

    This involves tasks where the LLM is expected to produce creative, coherent, and contextually
    relevant textual content based on the given input.

    Attributes:
        instructions (str): Instruction to guide the LLM in text generation.
    """

    name: str = "text_generation"
    instructions: str = "Generate text based on the provided input."
    input_template: str = "Input: {text}"
    output_template: str = "Output: {generated_text}"
