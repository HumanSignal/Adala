from pydantic import model_validator
from typing import Optional
from adala.skills._base import TransformSkill
from adala.utils.internal_data import InternalDataFrame
from adala.runtimes.base import Runtime
from adala.memories import Memory
from adala.memories.vectordb import VectorDBMemory


class RAGSkill(TransformSkill):
    """
    Skill for RAG (Retrieval-Augmented Generation) models.

    Attributes:
        input_template: Template for the input. It wraps the input with the template to create a query.
        rag_input_template: Template for RAG input. It wraps each retrieved item with the template, and then concatenates them with two newlines.
                            Example: "Question: {question}\nContext: {context}" with num_results=2 will result in "Question: <question>\nContext: <context>\n\nQuestion: <question>\nContext: <context>"
        instructions: Instructions for the generation part of the RAG model.
        output_template: Template for the output. It wraps the output with the template.
        num_results: Number of results to retrieve from the memory.
        memory: Memory to use for retrieval. If None, a VectorDBMemory will be used.

    Examples:
        >>> from adala.skills import RAGSkill
        >>> skill = RAGSkill(
        ...     name="rag",
        ...     input_template="Question: {question}",
        ...     rag_input_template="Question: {question}\nContext: {context}",
        ...     instructions="Answer the question.",
        ...     output_template="{answer}",
        ...     num_results=2,
        ... )
        >>> skill.apply(
        ...     input=InternalDataFrame(
        ...         data=[
        ...             {"question": "What is the meaning of life?", "context": "Life is a game."},
        ...             {"question": "What is the meaning of life?", "context": "Life is a game."},
        ...         ]
        ...     ))

    """

    name: str = "rag"
    rag_input_template: str
    instructions: str = ""
    output_template: str = "{rag}"
    num_results: int = 1
    memory: Memory = None

    @model_validator(mode="after")
    def init_memory(self):
        if self.memory is None:
            self.memory = VectorDBMemory(db_name=self.name)
        return self

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:
        """
        Apply the skill.

        Args:
            input: Input data.
            runtime: Runtime to use for generation.

        Returns:
            Output data. The output field is named after the output_template.
                        If no instructions are given, the output field contains concatenated strings from retrieved items.
                        If instructions are given, the output field contains the generated output.
        """
        input_strings = input.apply(
            lambda r: self.input_template.format(**r), axis=1
        ).tolist()
        rag_input_data = self.memory.retrieve_many(
            input_strings, num_results=self.num_results
        )
        rag_input_strings = [
            "\n\n".join(self.rag_input_template.format(**i) for i in rag_items)
            for rag_items in rag_input_data
        ]
        output_fields = self.get_output_fields()
        if len(output_fields) != 1:
            raise ValueError(
                f"RAG skill must have exactly one output field, but has {len(output_fields)}"
            )
        output_field = output_fields[0]
        rag_input = InternalDataFrame({output_field: rag_input_strings})
        if self.instructions:
            # if instructions are given, use the runtime to generate the output
            output = runtime.batch_to_batch(
                rag_input,
                instructions_template=self.instructions,
                input_template=f"{{{output_field}}}",
                output_template=self.output_template,
            )
        else:
            # if no instructions - simply return the rag input
            output = rag_input
            output.index = input.index

        return output

    def improve(
        self,
        predictions: InternalDataFrame,
        train_skill_output: str,
        feedback,
        runtime: Runtime,
    ):
        """
        Improve the skill by storing the feedback match errors in the memory.

        Args:
            predictions: Predictions made by the skill.
            train_skill_output: Output field of the skill used for training.
            feedback: Feedback data. for feedback.match equals False (prediction errors), the input is stored in the memory.
            runtime: Runtime to use for generation (not used).
        """

        error_indices = feedback.match[
            (feedback.match.fillna(True) == False).any(axis=1)
        ].index
        inputs = predictions.loc[error_indices]
        input_strings = inputs.apply(
            lambda r: self.input_template.format(**r), axis=1
        ).tolist()
        fb = feedback.feedback.loc[error_indices].rename(columns=lambda c: f"{c}__fb")
        inputs = inputs.join(fb)
        self.memory.remember_many(input_strings, inputs.to_dict(orient="records"))
