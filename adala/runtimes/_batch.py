from tqdm import tqdm
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.runtimes.base import Runtime
from typing import Optional, Dict, Any, List
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
from pydantic import Field
from adala.utils.parse import parse_template, partial_str_format
from adala.utils.matching import match_options
from ._openai import chat_completion_call


class BatchRuntime(Runtime):

    runtime_model: str = Field(alias="model", default_factory=lambda: "mistralai/Mistral-7B-Instruct-v0.2")

    _llm = None
    _tokenizer = None
    _max_tokens = None

    def init_runtime(self) -> "Runtime":
        if _VLLM_AVAILABLE:
            self._init_vllm()
        else:
            self._init_vanilla()
        return self

    def _init_vllm(self):
        self._llm = LLM(model=self.runtime_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.runtime_model)

    def _init_vanilla(self):
        pass

    def _convert(self, string):
        return self._tokenizer.apply_chat_template([{'role': 'user', 'content': string}], tokenize=False)

    def execute(self, prompts, options):
        if not _VLLM_AVAILABLE:
            # fallback to vanilla OpenAI API calls
            completions = []
            for prompt in tqdm(prompts):
                messages = [{'role': 'user', 'content': prompt}]
                if self.verbose:
                    print(f'Prompt: {prompt}')
                completion = chat_completion_call('gpt-4-1106-preview', messages)
                completion_text = completion.choices[0].message.content
                if options:
                    completion_option = match_options(completion_text, options)
                    if self.verbose:
                        print(f'Completion text: {completion_text}, matched option: {completion_option}')
                    completions.append(completion_option)
                else:
                    if self.verbose:
                        print(f'Completion text: {completion_text}')
                    completions.append(completion_text)
            return completions

        if not self._max_tokens:
            self._max_tokens = max(map(lambda o: len(self._tokenizer.tokenize(o)), options))
        params = SamplingParams(max_tokens=self._max_tokens)
        prepared_prompts = map(self._convert, prompts)
        outputs = self._llm.generate(prepared_prompts, params)
        completions = []
        for output in outputs:
            completion = output.outputs[0].text
            if options:
                completion = match_options(completion, options)
            completions.append(completion)
        return completions

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> Dict[str, str]:
        raise NotImplementedError

    def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        options: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        extra_fields = extra_fields or {}
        input_template = partial_str_format(input_template, **extra_fields)
        output_template = partial_str_format(output_template, **extra_fields)
        output_fields = parse_template(output_template, include_texts=True)

        if instructions_first:
            tmpl = f'{instructions_template}\n\n{input_template}\n'
        else:
            tmpl = f'{input_template}\n\n{instructions_template}\n\n'

        df_completions = InternalDataFrame()
        options = options or {}
        for output_field in output_fields:
            if output_field['type'] == 'text':
                tmpl += output_field['text']
            elif output_field['type'] == 'var':
                output_name = output_field['text']
                prompts = InternalDataFrameConcat((batch, df_completions), axis=1).apply(lambda r: tmpl.format(**r), axis=1)
                completions = self.execute(prompts, options=options.get(output_name))

                df_completions = InternalDataFrameConcat(
                    (df_completions, InternalDataFrame(data=completions, index=batch.index, columns=[output_name])),
                    axis=1
                )
                tmpl += f'{{{output_name}}}'

        return df_completions