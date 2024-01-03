from utils import patching, PatchedCalls


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[
        {
            "input": {},
            "output": {
                "data": [{"id": "gpt-3.5-turbo"}, {"id": "gpt-3.5-turbo-instruct"}]
            },
        }
    ],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        {
            "input": {
                "input": 'Input JSON format: {"a": 1, "b": 2}\nInput JSON format: {"a": 3, "b": 4}\nInput JSON format: {"a": 5, "b": 6}'
            },
            "output": {"code": "def convert(input_json):\npass"},
        }
    ],
)
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
                    "code": "def convert(input_json):\npass",
                }
            ]
        ),
    )
