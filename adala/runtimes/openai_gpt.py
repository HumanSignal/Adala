import openai

from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_random
from .base import LLMRuntime


@retry(wait=wait_random(min=5, max=10), stop=stop_after_attempt(6))
def _openai_completion_create_with_retry(**kwargs) -> Dict:
    return openai.Completion.create(**kwargs)


class OpenAIGPTRuntime(LLMRuntime):

    openai_model_name: str = 'gpt-3.5-turbo-instruct'
    openai_temperature: float = 0
    openai_max_tokens: int = 10
    verbose: bool = False

    def process_batch(self, batch: List[str]) -> List[Dict]:
        completions = _openai_completion_create_with_retry(
            model=self.openai_model_name,
            prompt=batch,
            max_tokens=self.openai_max_tokens,
            temperature=self.openai_temperature,
            logprobs=1
        )
        return [{
            self.text_key: c['text'],
            self.score_key: sum(c['logprobs']['token_logprobs']) / len(c['logprobs']['token_logprobs']),
            # TODO: add rationale
            self.rationale_key: c['text']
        } for c in completions['choices']]
