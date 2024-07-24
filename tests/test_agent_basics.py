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

    # assert final instruction
    # TODO this doesn't look right, investigate
    # seems the teacher prompt is in this field instead of the student prompt
    # assert agent.skills["0_to_1"].instructions == 'Refine the prompt to address the issues raised in the user feedback:\nApply the calculation rule to each number in the set as follows: add 1 to the first and third numbers while keeping the second number unchanged. If no specific rule is provided, use the default rule mentioned.'


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
                model='gpt-4-turbo-preview', max_tokens=4096, temperature=None
            )
        },
    )

    agent.learn()

    # assert final instruction
    assert (
        agent.skills["0->1"].instructions
        == "Transform each sequence of numbers provided as input by replacing all '0's with '1's, and leaving all other numbers unchanged to generate the modified sequence. Ensure the output presents only the sequence of numbers directly, without repeating any additional text or format from the input."
    )
    assert (
        agent.skills["1->2"].instructions
        == 'For each sequence of numbers given, increment each number in the input by 1, maintaining their order and output the modified sequence accordingly.'
    )
