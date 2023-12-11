from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.runtimes.base import Runtime
from typing import Optional, Dict, Any
from vllm import LLM, SamplingParams
from pydantic import Field
from adala.utils.parse import parse_template, partial_str_format


class VLLMOfflineRuntime(Runtime):

    vllm_model: str = Field(alias="model")

    _llm = None

    def init_runtime(self) -> "Runtime":
        self._llm = LLM(model=self.vllm_model)
        return self

    def execute(self, prompts, **kwargs):
        params = SamplingParams(**kwargs)
        outputs = self._llm.generate(prompts, params)
        generated_texts = []
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
        return generated_texts

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
        field_schema: Optional[Dict] = None,
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
        for output_field in output_fields:
            if output_field['type'] == 'text':
                tmpl += output_field['text']
            elif output_field['type'] == 'var':
                output_name = output_field['text']
                prompts = InternalDataFrameConcat((batch, df_completions), axis=1).apply(lambda r: tmpl.format(**r), axis=1)
                completions = self.execute(prompts, stop='\n', max_tokens=16)
                df_completions = InternalDataFrameConcat(
                    (df_completions, InternalDataFrame(data=completions, index=batch.index, columns=[output_name])),
                    axis=1
                )
                tmpl += f'{{{output_name}}}'
        return df_completions
