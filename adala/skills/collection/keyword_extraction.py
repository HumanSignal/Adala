from adala.skills._base import TransformSkill
from typing import List, Dict, Optional
from pydantic import model_validator, BaseModel
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.runtimes import Runtime
from adala.utils.logs import print_error


class KeywordExtractionSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    """

    class LabelItem(BaseModel):
        name: str
        instruction: str  # instruction on how to use this label

    name: str = "keyword_extraction"
    instructions: str = '''
## LABELS    
    
## Instructions:

1. Read the Provided Text Thoroughly: Carefully examine the entire content of the text you are given.
2. Identify Keywords: Detect and note down the keywords within the text. Keywords are defined as words or phrases that carry significant meaning or relevance in relation to the context of the text and are directly associated with the LABELS specified.
3. Preserve Original Order and Spelling: List the identified keywords in the exact sequence they occur within the text, maintaining their original spelling and capitalization.
4. Assign Labels to Keywords: For each identified keyword, you must allocate an appropriate label from the LABELS listed. Each label should closely match the context or category the keyword pertains to.
5. Explanation: For each identified keyword, you must write a short explanation why you used this label.
6. Handle Multiple Occurrences: If a keyword is found multiple times in the text, and its context or form varies, you should list and label each instance separately, treating them as distinct entries.
7. Format Your Response: Present the keywords and their respective labels in the format 'keyword-or-phrase // label // explanation'. Your response should solely consist of a list of these entries, each on a new line, arranged in the order they appear within the text.
8. Exclusive Output Requirement: Your output should exclusively contain the list of identified keywords and their corresponding labels, formatted as instructed. Exclude any additional text or commentary from your response.
9. Label Format: LABELS can include per-label instructions on how to use this label in format: <number_of_label>. "<label_name>" - <instructions>. 
10. If no label is assigned to a keyword, you don't need to list this keyword. 
11. If two or more keywords go together, you can stack them to keyword phrase and label together.  
'''
    input_template: str = 'Input:\n"""\n{text}\n"""'
    output_template: str = "Output:\n{keywords}"
    labels: Optional[Dict[str, List[LabelItem]]]

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
        labels = self.labels[output_fields[0]]
        labels_list = "\n".join([
            f'{i}. "{label.name}"' +
            (" - " + label.instruction + '.') if label.instruction else ''
            for i, label in enumerate(labels)
        ])
        self.instructions = self.instructions.replace('## LABELS', '## LABELS\n' + labels_list)

    def _postprocess_output(self, row, output_field: str) -> list:
        labels = [label.name for label in self.labels[output_field]]
        output_string = row[output_field]
        text = row["text"]

        # given the output string, extract corresponding positions of the words in the input text and return them

        start_index = 0
        output = []
        # print('==>', text, '\n', output_string)

        for line in output_string.split('\n'):
            if not line:
                continue

            split = line.split('//')
            if len(split) != 3:
                print_error(f"Keyword and label couldn't be parsed: {line}")
                continue

            keyword, label, explain = split[0].strip(), split[1].strip(), split[2].strip()
            if label not in labels:
                print_error(f"Label '{label}' not in provided labels: {labels}")
                continue

            index = text.find(keyword, start_index)
            if index != -1:
                output.append({
                    'start': index,
                    'end': index + len(keyword),
                    'labels': [label],
                    'text': keyword,
                    'explain': explain
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

        # apply postprocessing with keyword extraction from output
        for output_field in self.get_output_fields():
            batch[output_field] = input_batch.apply(
                self._postprocess_output,
                output_field=output_field,
                axis=1
            )

        return batch

