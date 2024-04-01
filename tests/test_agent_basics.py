import pandas as pd

from utils import patching, PatchedCalls, OpenaiChatCompletionMock


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_RETRIEVE.value,
    data=[
        # calling API model list for main runtime
        {"input": {}, "output": {}},
        # calling API model list for teacher runtime
        {"input": {}, "output": {}}
    ],
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        # call[0]: Iteration#1: apply skill 0->1, first row, GT = 1 5 1
        {
            "input": {"model": "gpt-3.5-turbo", "messages":
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "Input: 0 5 0\nOutput: "}]},
            "output": OpenaiChatCompletionMock('1 5 1'),
        },
        # call[1]: Iteration#1: apply skill 0->1, second row, GT = 1 1 1 -> ERROR!
        {
            "input": {"model": "gpt-3.5-turbo", "messages":
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "Input: 0 0 0\nOutput: "}]},
            "output": OpenaiChatCompletionMock('1 5 1'), # ERROR, should be 1 1 1
        },
        # call[2]: Learning phase: reasoning
        {
            "input": {'model': 'gpt-3.5-turbo'},
            "output": OpenaiChatCompletionMock(content="Reasoning: I should transform 0 to 1"),
        },
        # call[3]: Learning phase: generate new instruction
        {
            "input": {'model': 'gpt-3.5-turbo'},
            "output": OpenaiChatCompletionMock(content="Transform 0 to 1"),
        },
        # call[4]: Iteration#2: reapply skill 0->1, first row
        {
            "input": {"model": "gpt-3.5-turbo", "messages":
                [{"role": "system", "content": "Transform 0 to 1"},
                 {"role": "user", "content": "Input: 0 5 0\nOutput: "}]},
            "output": OpenaiChatCompletionMock('1 5 1'),
        },
        # call[5]: Iteration#2: reapply skill 0->1, second row
        {
            "input": {"model": "gpt-3.5-turbo", "messages":
                [{"role": "system", "content": "Transform 0 to 1"},
                 {"role": "user", "content": "Input: 0 0 0\nOutput: "}]},
            "output": OpenaiChatCompletionMock('1 1 1'), # Now correct
        },
    ],
    strict=False,
)
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
        teacher_runtimes={"default": OpenAIChatRuntime(model='gpt-3.5-turbo')}
    )
    assert agent.skills.get_skill_outputs() == {"0_to_1": "0_to_1"}

    agent.learn(learning_iterations=2)

    # assert final instruction
    assert agent.skills["0_to_1"].instructions == "Transform 0 to 1"


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_RETRIEVE.value,
    data=[
        # calling API model list for main runtime
        {"input": {}, "output": {}},
        # calling API model list for teacher runtime
        {"input": {}, "output": {}}
    ],
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        # call[0]: apply first skill 0->1, first row "0 5 0", output "1 5 1", GT = 1 5 1
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "Text: 0 5 0\nOutput: "}]},
         "output": OpenaiChatCompletionMock("1 5 1")},
        # # call[1]: apply first skill 0->1, second row "0 0 0", output "1 5 1", GT = 1 1 1 -> ERROR!
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "Text: 0 0 0\nOutput: "}]},
         "output": OpenaiChatCompletionMock("1 5 1")},
        # # call[2]: apply second skill 1->2, 1st skill first row "1 5 1", output "2 5 2", GT = 2 5 2
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "Text: 1 5 1\nOutput: "}]},
         "output": OpenaiChatCompletionMock("2 5 2")},
        # # call[3]: apply second skill 1->2, 1st skill second row "1 5 1", output "2 5 2", GT = 2 2 2 -> ERROR
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "Text: 1 5 1\nOutput: "}]},
         "output": OpenaiChatCompletionMock("2 5 2")},
        # Learning phase: reasoning and instruction generation calls
        # call[4]
        {
            "input": {'model': 'gpt-4-turbo-preview'},
            "output": OpenaiChatCompletionMock(content="Reasoning: I should transform 0 to 1"),
        },
        # call[5]
        {
            "input": {'model': 'gpt-4-turbo-preview'},
            "output": OpenaiChatCompletionMock(content="Transform 0 to 1"),
        },
        # call[6]: reapply first skill 0->1, first row "0 5 0", output "1 5 1", GT = 1 5 1
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "Transform 0 to 1"},
             {"role": "user", "content": "Text: 0 5 0\nOutput: "}]},
         "output": OpenaiChatCompletionMock("1 5 1")},
        # call[7]: reapply first skill 0->1, second row "0 0 0", output "1 1 1", GT = 1 1 1
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "Transform 0 to 1"},
             {"role": "user", "content": "Text: 0 0 0\nOutput: "}]},
         "output": OpenaiChatCompletionMock("1 1 1")},
        # call[8]: reapply second skill 1->2, 1st skill first row "1 5 1", output "2 2 2", GT = 2 5 2 -> ERROR!
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "Text: 1 5 1\nOutput: "}]},
         "output": OpenaiChatCompletionMock("2 2 2")},
        # call[9]: apply second skill 1->2, 1st skill second row "1 1 1", output "2 2 2", GT = 2 2 2
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "..."},
             {"role": "user", "content": "Text: 1 1 1\nOutput: "}]},
         "output": OpenaiChatCompletionMock("2 2 2")},
        # Learning phase for 2nd skill 1->2, since now it contains error for the first row
        # call[10]
        {
            "input": {'model': 'gpt-4-turbo-preview'},
            "output": OpenaiChatCompletionMock(content="Reasoning: I should transform 1 to 2"),
        },
        # call[11]
        {
            "input": {'model': 'gpt-4-turbo-preview'},
            "output": OpenaiChatCompletionMock(content="Transform 1 to 2"),
        },
        # And run again
        # call[12]: reapply first skill 0->1, first row "0 5 0", output "1 5 1", GT = 1 5 1
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "Transform 0 to 1"},
             {"role": "user", "content": "Text: 0 5 0\nOutput: "}]},
         "output": OpenaiChatCompletionMock("1 5 1")},
        # call[13]: reapply first skill 0->1, second row "0 0 0", output "1 1 1", GT = 1 1 1
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "Transform 0 to 1"},
             {"role": "user", "content": "Text: 0 0 0\nOutput: "}]},
         "output": OpenaiChatCompletionMock("1 1 1")},
        # call[14]: reapply second skill 1->2, 1st skill first row "1 5 1", output "2 5 2", GT = 2 5 2
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "Transform 1 to 2"},
             {"role": "user", "content": "Text: 1 5 1\nOutput: "}]},
         "output": OpenaiChatCompletionMock("2 5 2")},
        # call[15]: apply second skill 1->2, 1st skill second row "1 1 1", output "2 2 2", GT = 2 2 2
        {"input": {"model": "gpt-3.5-turbo", "messages":
            [{"role": "system", "content": "Transform 1 to 2"},
             {"role": "user", "content": "Text: 1 1 1\nOutput: "}]},
         "output": OpenaiChatCompletionMock("2 2 2")},
    ],
)
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
                [
                    ["0 5 0", "1 5 1", "2 5 2"],
                    ["0 0 0", "1 1 1", "2 2 2"]
                ],
                columns=["input", "gt_0", "gt_1"],
            ),
            ground_truth_columns={"0->1": "gt_0", "1->2": "gt_1"},
        ),
        teacher_runtimes={
            'default': OpenAIChatRuntime(model='gpt-4-turbo-preview')
        }
    )

    agent.learn()

    # assert final instruction
    assert agent.skills["0->1"].instructions == "Transform 0 to 1"
    assert agent.skills["1->2"].instructions == "Transform 1 to 2"
