from ..base import LLMSkill


class TextGenerationSkill(LLMSkill):
    instructions: str = 'Generate text based on the provided input.'
