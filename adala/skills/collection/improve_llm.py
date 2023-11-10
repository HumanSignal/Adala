from adala.skills._base import AnalysisSkill


class ImproveLLMInstructions(AnalysisSkill):
    """
    Improves LLM instructions given the error analysis report of the previous instruction
    """
    old_instructions: str
    name: str = 'improve_llm_instructions'
    instructions: str = "LLM prompt was created by concatenating instructions with text input:\n\n" \
                        "Prediction = LLM(Input, Instructions)\n\n" \
                        "We expect the prediction to be equal to the ground truth.\n" \
                        "Your task is to analyze errors made by old instructions " \
                        "and craft new instructions for the LLM.\n" \
                        "Follow best practices for LLM prompt engineering.\n" \
                        "Include 2-3 examples at the end of your response " \
                        "to demonstrate how the new instruction would be applied.\n" \
                        "Use the following format for your examples:\n" \
                        "Input: ...\n" \
                        "Output: ...\n\n"

    input_template: str = "Old instructions:\n{old_instructions}\n\n" \
                          "Errors:\n{error_report}\n\n"
    output_template: str = "New instructions:\n{new_instructions}\n\n"
