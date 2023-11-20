from utils import patching, PatchedCalls


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[{'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo'}, {'id': 'gpt-3.5-turbo-instruct'}]}}],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[{
        'input': {
            'input': 'Input JSON format: {"a": 1, "b": 2}\nInput JSON format: {"a": 3, "b": 4}\nInput JSON format: {"a": 5, "b": 6}'
        },
        'output': {
            'code': 'def convert(input_json):\npass\n'
        }
    }]
)
def test_code_generation():
    from adala.skills import AnalysisSkill, ParallelSkillSet
    from adala.agents import Agent
    from adala.environments import StaticEnvironment
    import pandas as pd

    env = StaticEnvironment(df=pd.DataFrame(
        {'payload': ['{"a": 1, "b": 2}', '{"a": 3, "b": 4}', '{"a": 5, "b": 6}']}))

    skillset = ParallelSkillSet(skills=[
        AnalysisSkill(
            name='code_generation',
            input_template="Input JSON format: {payload}",
            output_template="Code: {code}",
            instructions="Generate Python code that takes the input JSON and returns the output JSON"
        ),
    ])

    agent = Agent(skills=skillset, environment=env)
    predictions = agent.run()
    pd.testing.assert_frame_equal(
        predictions,
        pd.DataFrame([
            {'payload': '{"a": 1, "b": 2}', 'code': 'def convert(input_json):\npass\n'},
            {'payload': '{"a": 3, "b": 4}', 'code': 'def convert(input_json):\npass\n'},
            {'payload': '{"a": 5, "b": 6}', 'code': 'def convert(input_json):\npass\n'}
        ])
    )