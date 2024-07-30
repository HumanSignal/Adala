import pytest


@pytest.mark.vcr
def test_code_generation():
    from adala.skills import AnalysisSkill, LinearSkillSet  # type: ignore
    from adala.agents import Agent  # type: ignore
    from adala.environments import StaticEnvironment  # type: ignore
    import pandas as pd

    env = StaticEnvironment(
        df=pd.DataFrame(
            {"payload": ['{"a": 1, "b": 2}', '{"a": 3, "b": 4}', '{"a": 5, "b": 6}']}
        )
    )

    skillset = LinearSkillSet(
        skills=[
            AnalysisSkill(
                name="code_generation",
                input_template="Input JSON format: {payload}",
                output_template="Code: {code}",
                instructions="Generate Python code that calculates the sum of the given values per each key",
            ),
        ]
    )

    agent = Agent(skills=skillset, environment=env)
    predictions = agent.run()
    expected_code = """\
import json
from collections import defaultdict

def sum_values(json_list):
    result = defaultdict(int)
    for json_str in json_list:
        data = json.loads(json_str)
        for key, value in data.items():
            result[key] += value
    return dict(result)

# Example usage
input_jsons = [
    '{"a": 1, "b": 2}',
    '{"a": 3, "b": 4}',
    '{"a": 5, "b": 6}'
]

output = sum_values(input_jsons)
print(output)  # Output will be {'a': 9, 'b': 12}"""
    assert predictions.code[0] == expected_code
