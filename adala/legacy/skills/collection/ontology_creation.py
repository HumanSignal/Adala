from adala.skills._base import AnalysisSkill
from pydantic import model_validator


class OntologyCreator(AnalysisSkill):
    target: str
    name: str = "onto_creator"
    instructions: str = (
        """
Given the list of examples separated by '## Example <id>', assign not more than 5 categories that best describe the target for the given examples.
Don't repeat semantically similar categories if a single category can describe multiple examples.
Output only a list of assigned categories separated by a newline; don't output any additional text.""",
    )
    input_template: str = """
## Example {i}
Text: {text}\n\n"""
    output_template: str = "{categories}"
    input_separator: str = "\n"
    chunk_size: int = 200

    @model_validator(mode="after")
    def validate_target(self):
        self.instructions = f"The target is {self.target}.\n{self.instructions}"


class OntologyMerger(OntologyCreator):
    name: str = "onto_merger"
    instructions: str = '''
You will be given different groups of labels that fit this target, separated by '## Group <id>'. For example:
"""
## Group 1
label A
label B
label C
...

## Group 2
label similar to label A
label D
...


## Group 3
label similar to label D
label similar to label B
label A
...
"""

Your goal is to squash all presented groups into a single group of labels that merges all semantically similar labels into one more generic label.
For example, some labels from the three groups can be merged into the generic labels, since similar labels are presented in different groups:

"""
generic label A
generic label B
label C
generic label D
...
"""

When creating the output list of labels, please follow the guideline:

1. Please use only the label names that are well understood and represent general categories to comply with the initial target.
2. Try to create distinct categories, don't output two categories if they represent two semantically similar closing lost reasons.
3. Don't try to merge label names if they represent different aspects of a given target - do it only if they represent the same meaning.
4. Don't output additional text, only general label names separated by newlines.
5. Don't ignore any labels presented in each group - every input label in groups must belong to a single output category.
6. Create up to 10 different categories.
'''
    input_template: str = "## Group {i}\n{categories}\n\n"
    output_template: str = "Output categories: {labels}"
