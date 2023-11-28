from typing import Optional
from adala.skills._base import TransformSkill
from adala.utils.internal_data import InternalDataFrame
from adala.runtimes.base import Runtime
from adala.memories import Memory


class RAGSkill(TransformSkill):
    """
    Skill for RAG (Retrieval-Augmented Generation) models.
    """
    name: str = 'rag'
    rag_input_template: str
    instructions: str = ''
    output_template: str = '{rag}'
    num_results: int = 1
    memory: Memory

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Apply the skill.
        """
        input_strings = input.apply(lambda r: self.input_template.format(**r), axis=1).tolist()
        rag_input_data = self.memory.retrieve_many(input_strings, num_results=self.num_results)
        rag_input_strings = ['\n\n'.join(self.rag_input_template.format(**i) for i in rag_items) for rag_items in rag_input_data]
        output_fields = self.get_output_fields()
        if len(output_fields) != 1:
            raise ValueError(f'RAG skill must have exactly one output field, but has {len(output_fields)}')
        output_field = output_fields[0]
        rag_input = InternalDataFrame({output_field: rag_input_strings})
        if self.instructions:
            output = runtime.batch_to_batch(
                rag_input,
                instructions_template=self.instructions,
                input_template=f'{{{output_field}}}',
                output_template=self.output_template
            )
        else:
            output = rag_input.reindex(input.index)
        return output

    def improve(
        self,
        predictions: InternalDataFrame,
        train_skill_output: str,
        feedback,
        runtime: Runtime,
    ):
        """
        Improve the skill.
        """

        error_indices = feedback.match[(feedback.match.fillna(True) == False).any(axis=1)].index
        inputs = predictions.loc[error_indices]
        input_strings = inputs.apply(lambda r: self.input_template.format(**r), axis=1).tolist()
        fb = feedback.feedback.loc[error_indices].rename(columns=lambda c: f'{c}__fb')
        inputs = inputs.join(fb)
        self.memory.remember_many(input_strings, inputs.to_dict(orient='records'))
