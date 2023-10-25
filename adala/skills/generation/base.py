from ..base import LLMSkill


class TextGenerationSkill(LLMSkill):
    """
    Skill specialized for generating text based on the provided input.

    This involves tasks where the LLM is expected to produce creative, coherent, and contextually 
    relevant textual content based on the given input.

    Attributes:
        instructions (str): Instruction to guide the LLM in text generation.
    """
    
    instructions: str = 'Generate text based on the provided input.'
