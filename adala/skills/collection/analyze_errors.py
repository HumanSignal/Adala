from adala.skills._base import AnalysisSkill
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.utils.parse import parse_template, partial_str_format
from adala.runtimes.base import Runtime
from typing import Dict


class AnalyzeLLMPromptErrorsExactMatch(AnalysisSkill):
    """
    Analyzes errors in a text.
    """
    name: str = 'analyze_llm_prompt_errors_exact_match'
    initial_llm_instructions: str
    instructions: str = "LLM prompt was created by concatenating instructions with text input:\n\n" \
                        "Prediction = LLM(Input, Instructions)\n\n" \
                        "We expect the prediction to be equal to the ground truth.\n" \
                        "Your task is to provide a reason for the error due to the original instruction.\n" \
                        "Be concise and specific. The reason for the error should fit within a single line.\n\n" \
                        "Instructions:\n{initial_llm_instructions}\n\n"
    prediction_column: str
    ground_truth_column: str

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> Dict[str, str]:
        """
        Applies the skill to a record and returns a dataframe.

        Args:
            input (InternalDataFrame): The input data to be processed.
            runtime (Runtime): The runtime instance to be used for processing.
        """

        output_fields = parse_template(self.output_template, include_texts=False)
        if len(output_fields) > 1:
            raise ValueError(f'Output template should contain only one field, got {output_fields}')
        output_field_name = output_fields[0]['text']

        MAX_ERRORS = 3
        errors = input.sample(n=min(MAX_ERRORS, input.shape[0]))

        input_template = f'{self.input_template}\n' \
                         f'Prediction: {{{self.prediction_column}}}\n' \
                         f'Ground truth: {{{self.ground_truth_column}}}\n'
        output_template = 'Error reason: {reason}\n'

        errors_with_reason = runtime.batch_to_batch(
            batch=errors,
            input_template=input_template,
            output_template=output_template,
            instructions_template=self.instructions,
            extra_fields={'initial_llm_instructions': self.initial_llm_instructions},
            field_schema=self.field_schema,
        )

        errors_with_reason = InternalDataFrameConcat([errors, errors_with_reason], axis=1)

        agg_template = f'{input_template}{output_template}'
        aggregated_errors = errors_with_reason.apply(
            lambda row: agg_template.format(**row), axis=1
        ).str.cat(sep='\n')

        return {output_field_name: aggregated_errors}
