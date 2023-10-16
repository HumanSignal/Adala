import guidance
import enum

from tqdm import tqdm
from typing import List, Dict
from .base import LLMRuntime


class LabelsConstrainedRuntime(LLMRuntime):
    labels: List[str]


class GuidanceModelTypes(enum.Enum):
    OpenAI = 'OpenAI'
    Transformers = 'Transformers'


class GuidanceRuntime(LabelsConstrainedRuntime):

    guidance_template: str = '''\
{{{{prefix}}}}

Describe your reasoning step-by-step then provide your output.

Reasoning: {{{{gen '{rationale}'}}}}
Output: {{{{select '{text}' options=labels logprobs='{score}'}}}}
'''
    guidance_llm_type: GuidanceModelTypes = GuidanceModelTypes.OpenAI
    guidance_llm_params: Dict[str, str] = {
        'model': 'gpt-3.5-turbo-instruct'
    }
    llm: guidance.llms.LLM = None
    _program = None

    class Config:
        arbitrary_types_allowed = True

    def init_runtime(self):
        if not self.llm:
            # create an LLM instance
            if self.guidance_llm_type == GuidanceModelTypes.OpenAI:
                self.llm = guidance.llms.OpenAI(**self.guidance_llm_params)
            elif self.guidance_llm_params == GuidanceModelTypes.Transformers:
                self.llm = guidance.llms.Transformers(**self.guidance_llm_params)

        self.guidance_template = self.guidance_template.format(
            rationale=self.rationale_key,
            text=self.text_key,
            score=self.score_key
        )

        self._program = guidance(self.guidance_template, llm=self.llm)
        return self

    def process_batch(self, batch: List[str]) -> List[Dict]:
        output = []
        for prefix in tqdm(batch, disable=not self.verbose, desc='Processing batch'):
            result = self._program(prefix=prefix, labels=self.labels, silent=True)
            output.append({
                self.text_key: result[self.text_key],
                self.score_key: result[self.score_key][result[self.text_key]],
                self.rationale_key: result[self.rationale_key]
            })
        return output
