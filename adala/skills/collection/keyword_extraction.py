from adala.skills._base import TransformSkill
from typing import List, Dict, Optional
from pydantic import model_validator, BaseModel
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.runtimes import Runtime
from adala.utils.parse import parse_template
from adala.utils.logs import print_error


class KeywordExtractionSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    """

    class LabelItem(BaseModel):
        name: str
        instruction: str  # Assuming instruction is a string. Adjust the type as needed.

    name: str = "keyword_extraction"
    instructions: str = '''
Given the following text, identify and extract the keywords. Assign a label for each keyword. \
It is crucial to extract the keywords exactly as they appear in the text, maintaining their 
original form and capitalization. \
If a word appears multiple times, each occurrence should be labeled in accordance with its 
specific context or form in the text. \

## Instructions:

1. Read the text thoroughly.
2. Identify the keywords in the text. Keywords are words that hold significant meaning or importance 
in the context of the text and related to LABELS only. 
3. Extract these words exactly as they appear, preserving their original  spelling and capitalization.
4. For each keyword, assign a label from the provided LABELS.
5. If a keyword appears multiple times in different contexts or forms, label each occurrence separately.
6. List the keywords from text (the same as they appear in the text) and their 
   corresponding labels in the order they appear in the text in the format 'keyword-or-phrase // label'.
7. Do not output any text except the newline separated list of keywords and corresponding labels.

## Example 
Note: LABELS will be different, it's just an example
Input: """Deep learning improves image processing. Learning models adapt to new images."""
Output:
Deep learning // Field
image processing // Task
Learning // Action
images // Object
'''
    input_template: str = 'Input:\n"""\n{text}\n"""'
    output_template: str = "Output:\n{keywords}"
    labels: Optional[Dict[str, List[Dict['name':str, 'instruction':str]]]]

    @model_validator(mode="after")
    def validate_labels(self):
        output_fields = self.get_output_fields()
        if len(output_fields) != 1:
            raise ValueError(
                f"{self.__class__.name} only supports one output field, got {output_fields}"
            )

        self.field_schema = {}
        for output_field, labels in self.labels.items():
            if output_field not in output_fields:
                raise ValueError(
                    f'Labels class "{output_field}" not in output fields {output_fields}'
                )

        # add label list to instructions
        # TODO: doesn't work for multiple outputs
        self.instructions += "\n## LABELS\nUse the following labels only:\n"
        labels_list = "\n".join(self.labels[output_fields[0]])
        self.instructions += f"{labels_list}\n\n"

    def _postprocess_output(self, row, output_field: str) -> str:
        output_string = row[output_field]
        text = row["text"]

        # given the output string, extract corresponding positions of the words in the input text and return them

        start_index = 0
        output = []
        print('==>', output_string)

        for line in output_string.split('\n'):
            split = line.split('//')
            if len(split) != 2:
                print_error(f"Keyword and label couldn't be parsed: {line}")
                continue

            keyword, label = split[0].strip(), split[1].strip()
            if label not in self.labels[output_field]:
                print_error(f"Label '{label}' not in provided labels: {self.labels[output_field]}")
                continue

            index = text.find(keyword, start_index)
            if index != -1:
                output.append({
                    'start': index,
                    'end': index + len(keyword),
                    'labels': [label]
                })
                start_index = index + len(keyword)
            else:
                print_error(f"Keyword '{keyword}' not found in text: {text}")

        return output

    def apply(
        self,
        input: InternalDataFrame,
        runtime: Runtime,
    ) -> InternalDataFrame:

        runtime.verbose = True
        batch = super(KeywordExtractionSkill, self).apply(input, runtime)
        input_batch = InternalDataFrameConcat((input, batch), axis=1)
        print(batch)

        # apply postprocessing with keyword extraction from output
        for output_field in self.get_output_fields():
            batch[output_field] = input_batch.apply(
                self._postprocess_output,
                output_field=output_field,
                axis=1
            )

        return batch

