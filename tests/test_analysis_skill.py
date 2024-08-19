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
                name="Output",
                input_template="Input JSON format: {payload}",
                output_template="Code: {code}",
                instructions="Generate Python code that calculates the sum of the given values per each key",
            ),
        ]
    )

    agent = Agent(skills=skillset, environment=env)
    predictions = agent.run()
    expected_code = """\
# Given input JSON format
input1 = {"a": 1, "b": 2}
input2 = {"a": 3, "b": 4}
input3 = {"a": 5, "b": 6}

# Calculate the sum of values per key
result = {}
for key in input1.keys():
    result[key] = input1[key] + input2[key] + input3[key]

result"""
    assert predictions.code[0] == expected_code
