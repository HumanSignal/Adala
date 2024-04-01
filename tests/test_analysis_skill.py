from utils import patching, PatchedCalls, OpenaiChatCompletionMock


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_RETRIEVE.value,
    data=[{"input": {}, "output": {}}],
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        {
            "input": {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "Generate Python code that takes the input JSON and returns the output JSON",
                    },
                    {
                        "role": "user",
                        "content": 'Input JSON format: {"a": 1, "b": 2}\n'
                        'Input JSON format: {"a": 3, "b": 4}\n'
                        'Input JSON format: {"a": 5, "b": 6}\n'
                        "Code: ",
                    },
                ],
            },
            "output": OpenaiChatCompletionMock("def convert(input_json):\npass"),
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
