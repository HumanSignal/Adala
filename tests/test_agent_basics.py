import pandas as pd
import pytest


@pytest.mark.vcr
def test_agent_quickstart_single_skill():
    from adala.agents import Agent  # type: ignore
    from adala.skills import LinearSkillSet, TransformSkill  # type: ignore
    from adala.environments import StaticEnvironment  # type: ignore
    from adala.runtimes import OpenAIChatRuntime  # type: ignore

    agent = Agent(
        skills=LinearSkillSet(
            skills=[
                TransformSkill(
                    name="0_to_1",
                    instructions="...",
                    input_template="Input: {input}",
                    output_template="Output: {0_to_1}",
                )
            ]
        ),
        environment=StaticEnvironment(
            df=pd.DataFrame(
                [["0 5 0", "1 5 1"], ["0 0 0", "1 1 1"]], columns=["input", "gt_0"]
            ),
            ground_truth_columns={"0_to_1": "gt_0"},
        ),
        teacher_runtimes={"default": OpenAIChatRuntime(model="gpt-3.5-turbo")},
    )
    assert agent.skills.get_skill_outputs() == {"0_to_1": "0_to_1"}

    agent.learn(learning_iterations=2)

    assert agent.skills["0_to_1"].instructions == 'Given a set of three numbers, increment each number by 1 individually to generate the output.'

@pytest.mark.vcr
def test_agent_quickstart_two_skills():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet, TransformSkill
    from adala.environments import StaticEnvironment
    from adala.runtimes import OpenAIChatRuntime

    agent = Agent(
        skills=LinearSkillSet(
            skills=[
                TransformSkill(
                    name="0->1",
                    instructions="...",
                    input_template="Text: {input}",
                    output_template="Output: {0->1}",
                ),
                TransformSkill(
                    name="1->2",
                    instructions="...",
                    input_template="Text: {0->1}",
                    output_template="Output: {1->2}",
                ),
            ]
        ),
        environment=StaticEnvironment(
            df=pd.DataFrame(
                [["0 5 0", "1 5 1", "2 5 2"], ["0 0 0", "1 1 1", "2 2 2"]],
                columns=["input", "gt_0", "gt_1"],
            ),
            ground_truth_columns={"0->1": "gt_0", "1->2": "gt_1"},
        ),
        teacher_runtimes={
            'default': OpenAIChatRuntime(
                model='gpt-4o-mini', max_tokens=4096, temperature=None
            )
        },
    )

    agent.learn()

    # assert final instruction
    assert (
        agent.skills["0->1"].instructions
        == '''\
Transform the set of three numbers represented as "X Y Z" into a new set of three numbers using the following transformation rules:

1. Change the first number to '1' if it is '0'; otherwise, it remains unchanged.
2. The middle number remains unchanged.
3. Change the third number to '1' if it is '0'; otherwise, it remains unchanged.

For the given input, the output must be in the format "A B C", where A, B, and C are the transformed numbers.

For example:
- If the input is "0 5 0", the correct output is "1 5 1".
- If the input is "0 0 0", the correct output is "1 1 1".

Please apply these transformation rules to provide the appropriate output for the input provided.'''
    )
    assert (
        agent.skills["1->2"].instructions
        == 'Transform the input sequence of three numbers by adding 1 to each number, and return the resulting sequence in the same format "<number1> <number2> <number3>".'
    )
