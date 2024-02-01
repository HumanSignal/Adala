from adala.skills._base import TransformSkill
from typing import List, Dict, Optional
from pydantic import model_validator
from adala.utils.internal_data import InternalDataFrame
from adala.runtimes import Runtime


class KeywordExtractionSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    """

    name: str = "keyword_extraction"
    instructions: str = '''
Given the following text, identify and extract the keywords. Assign a label for each keyword. \
It is crucial to extract the keywords exactly as they appear in the text, maintaining their original form and capitalization. \
If a word appears multiple times, each occurrence should be labeled in accordance with its specific context or form in the text. \

## Instructions:

1. Read the text thoroughly.
2. Identify the keywords in the text. Keywords are words that hold significant meaning or importance in the context of the text. Extract these words exactly as they appear, preserving their original spelling and capitalization.
3. For each keyword, assign a label from the provided list.
4. If a keyword appears multiple times in different contexts or forms, label each occurrence separately.
5. List the keywords and their corresponding labels in the order they appear in the text.
6. Separate extracted keywords and labels with double slash '//'.
7. Do not output any text except the newline separated list of keywords and corresponding labels.

## Example
Input: """Deep learning improves image processing. Learning models adapt to new images."""
Output:
Deep learning // Field
image processing // Task
Learning // Action
images // Object

'''
    input_template: str = 'Input:\n"""\n{text}\n"""'
    output_template: str = "Output:\n{keywords}"
    labels: Optional[Dict[str, List[str]]]

    @model_validator(mode="after")
    def validate_labels(self):
        output_fields = self.get_output_fields()
        if len(output_fields) != 1:
            raise ValueError(
                f"Classification skill only supports one output field, got {output_fields}"
            )
        self.field_schema = {}
        for labels_field, labels in self.labels.items():
            if labels_field not in output_fields:
                raise ValueError(
                    f'Labels class "{labels_field}" not in output fields {output_fields}'
                )

            self.field_schema[labels_field] = {
                "type": "array",
                "items": {"type": "string", "enum": labels},
            }

        # add label list to instructions
        # TODO: doesn't work for multiple outputs
        self.instructions += "\n\nAssume the following output labels:\n\n"
        labels_list = "\n".join(self.labels[output_fields[0]])
        self.instructions += f"{labels_list}\n\n"

    def _postprocess_output(self, row, output_field: str) -> str:
        output_string = row[output_field]
        # given the output string, extract corresponding positions of the words in the input text and return them
        text = row["text"]  # TODO: this should be the input field
        # TODO: @makseq: implement this
        return output_string

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:

        batch = super(KeywordExtractionSkill, self).apply(input, runtime)

        # apply postprocessing with keyword extraction from output
        for output_field in self.get_output_fields():
            batch[output_field] = batch.apply(
                self._postprocess_output,
                output_field=output_field,
                axis=1
            )

        return batch

