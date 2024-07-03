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
                instructions="Generate Python code that takes the input JSON and returns the output JSON",
            ),
        ]
    )

    agent = Agent(skills=skillset, environment=env)
    predictions = agent.run()
    pd.testing.assert_frame_equal(
        predictions,
        pd.DataFrame(
            [
                {
                    "code": '```python\nimport json\n\n# Input JSON\ninput_json = [\n    {"a": 1, "b": 2},\n    {"a": 3, "b": 4},\n    {"a": 5, "b": 6}\n]\n\n# Output JSON\noutput_json = json.dumps(input_json)\n\nprint(output_json)\n```'
                }
            ]
        ),
    )
