import openai

from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_random
from .base import LLMRuntime


@retry(wait=wait_random(min=5, max=10), stop=stop_after_attempt(6))
def _openai_completion_create_with_retry(model: str, prompt: List[str], max_tokens: int, temperature: float) -> Dict:
    return openai.Completion.create(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens)


class OpenAIGPTRuntime(LLMRuntime):

    model_name: str = 'gpt-3.5-turbo-instruct'
    temperature: float = 0
    max_tokens: int = 10
    verbose: bool = False

    def process_batch(self, batch: List[str]) -> List[str]:
        completions = _openai_completion_create_with_retry(
            model=self.model_name,
            prompt=batch,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return [c['text'] for c in completions['choices']]
