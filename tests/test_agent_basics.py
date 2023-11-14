import pandas as pd

from utils import patching, PatchedCalls, OpenaiChatCompletionMock


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[
        # calling API model list for the first runtime (student)
        {'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}, {'id': 'gpt-3.5-turbo'}, {'id': 'gpt-4'}]}},
        # calling API model list for the second runtime (teacher)
        {'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}, {'id': 'gpt-3.5-turbo'}, {'id': 'gpt-4'}]}},
    ],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        # call[0]: Iteration#1: apply skill 0->1, first row, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'0_to_1': '1 5 1'}},
        # call[1]: Iteration#1: apply skill 0->1, second row, GT = 1 1 1 -> ERROR!
        {'input': {'input': '0 0 0'}, 'output': {'0_to_1': '1 5 1'}},
        # call[2]: Iteration#2: reapply skill 0->1, first row
        {'input': {'input': '0 5 0'}, 'output': {'0_to_1': '1 5 1'}},
        # call[3]: Iteration#2: reapply skill 0->1, second row
        {'input': {'input': '0 0 0'}, 'output': {'0_to_1': '1 1 1'}},
    ]
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        {'input': {'model': 'gpt-3.5-turbo', 'messages': [
            {
                'role': 'system',
                'content': '''Act as LLM instructions generator. Full LLM prompt is created by concatenating LLM instructions and input text. You \
should respond only with the LLM instructions. After each generation, user provides a feedback which includes example of input text, LLM \
output, and user feedback. Based on this analysis, generate new instructions for the LLM. These instructions should be concise, direct, \
and focused solely on addressing the points raised in the user feedback, aligning with the input, and improving upon the predictions. \
Include relevant few-shot examples within the instructions that are aligned with the user's feedback and the initial input, \
demonstrating the desired format and approach for the LLM’s prediction. These examples should serve as clear models for the expected \
output in the next iteration.'''
            },
            {'role': 'assistant', 'content': '...'},
            {
                'role': 'user',
                'content': '''Here is the feedback based on the current instructions:\n\n\nInput: 0 5 0\nOutput: 1 5 1\nCorrect.\n\n\nInput: 0 0 \
0\nOutput: 1 5 1\nIncorrect. Must be equal to 1 1 1\n\n\n\nPlease address the feedback and provide new improved instructions for the \
LLM. Use the following format for the few-shot examples:\n\nInput: {input}\nOutput: {0_to_1}\n\nCarefully analyze this feedback, and \
provide updated prompting instructions for LLM:'''
            }
        ]},
            'output': OpenaiChatCompletionMock(content='Transform 0 to 1')}
    ],
    strict=True
)
def test_agent_quickstart_single_skill():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet, TransformSkill
    from adala.environments import StaticEnvironment

    agent = Agent(
        skills=LinearSkillSet(
            skills=[TransformSkill(
                name="0_to_1",
                instructions="...",
                input_template="Input: {input}",
                output_template="Output: {0_to_1}"
            )]
        ),
        environment=StaticEnvironment(
            df=pd.DataFrame([
                ['0 5 0', '1 5 1'],
                ['0 0 0', '1 1 1']
            ], columns=['input', 'gt_0']),
            ground_truth_columns={
                "0_to_1": "gt_0"
            }
        )
    )
    assert agent.skills.get_skill_outputs() == {'0_to_1': '0_to_1'}

    agent.learn(learning_iterations=2)

    # assert final instruction
    assert agent.skills['0_to_1'].instructions == 'Transform 0 to 1'


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[
        # calling API model list for the first runtime (student)
        {'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}, {'id': 'gpt-3.5-turbo'}, {'id': 'gpt-4'}]}},
        # calling API model list for the second runtime (teacher)
        {'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}, {'id': 'gpt-3.5-turbo'}, {'id': 'gpt-4'}]}},
    ],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        # call[0]: apply first skill 0->1, first row, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'0->1': '1 5 1'}},
        # call[1]: apply first skill 0->1, second row, GT = 1 1 1 -> ERROR!
        {'input': {'input': '0 0 0'}, 'output': {'0->1': '1 5 1'}},
        # call[2]: apply second skill 1->2, first row, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'1->2': '2 5 2'}},
        # call[3]: apply second skill 1->2, second row, GT = 2 2 2 -> ERROR
        {'input': {'input': '0 0 0', '0->1': '1 5 1'}, 'output': {'1->2': '2 5 2'}},
        # call[4]: reapply first skill 0->1, first row, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'0->1': '1 5 1'}},
        # call[5]: reapply first skill 0->1, second row, GT = 1 1 1
        {'input': {'input': '0 0 0'}, 'output': {'0->1': '1 1 1'}},
        # call[6]: reapply second skill 1->2, first row, GT = 2 5 2 -> ERROR!
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'1->2': '2 2 2'}},
        # call[7]: reapply second skill 1->2, second row, GT = 2 2 2
        {'input': {'input': '0 0 0', '0->1': '1 1 1'}, 'output': {'1->2': '2 2 2'}},
        # call[8]: reapply first skill 0->1, first row, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'0->1': '1 5 1'}},
        # call[9]: reapply first skill 0->1, second row, GT = 1 1 1
        {'input': {'input': '0 0 0'}, 'output': {'0->1': '1 1 1'}},
        # call[10]: reapply second skill 1->2, first row, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'1->2': '2 5 2'}},
        # call[11]: reapply second skill 1->2, second row, GT = 2 2 2
        {'input': {'input': '0 0 0', '0->1': '1 1 1'}, 'output': {'1->2': '2 2 2'}},
    ]
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        {'input': {}, 'output': OpenaiChatCompletionMock(content='Transform 0 to 1')},
        {'input': {}, 'output': OpenaiChatCompletionMock(content='Transform 1 to 2')},
    ])
def test_agent_quickstart_two_skills():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet, TransformSkill
    from adala.environments import StaticEnvironment

    agent = Agent(
        skills=LinearSkillSet(
            skills=[
                TransformSkill(name='0->1', instructions='...', input_template='Input: {input}', output_template='Output: {0->1}'),
                TransformSkill(name='1->2', instructions='...', input_template='Input: {0->1}', output_template='Output: {1->2}'),
            ]
        ),
        environment=StaticEnvironment(
            df=pd.DataFrame([
                ['0 5 0', '1 5 1', '2 5 2'],
                ['0 0 0', '1 1 1', '2 2 2']
            ], columns=['input', 'gt_0', 'gt_1']),
            ground_truth_columns={
                "0->1": "gt_0",
                "1->2": "gt_1"
            }
        )
    )

    agent.learn()

    # assert final instruction
    assert agent.skills['0->1'].instructions == 'Transform 0 to 1'
    assert agent.skills['1->2'].instructions == 'Transform 1 to 2'


@patching(
    target_function=PatchedCalls.OPENAI_MODEL_LIST.value,
    data=[
        # calling API model list for the first runtime (student)
        {'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}, {'id': 'gpt-3.5-turbo'}, {'id': 'gpt-4'}]}},
        # calling API model list for the second runtime (teacher)
        {'input': {}, 'output': {'data': [{'id': 'gpt-3.5-turbo-instruct'}, {'id': 'gpt-3.5-turbo'}, {'id': 'gpt-4'}]}},
    ],
)
@patching(
    target_function=PatchedCalls.GUIDANCE.value,
    data=[
        # call[0]: apply first skill 0->1, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'0->1': '1 5 1'}},
        # call[1]: apply second skill 1->2, GT = 2 5 2 -> ERROR!
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'1->2': '2 5 4'}},
        # call[2]: apply third skill 2->3, GT = 3 5 3 -> Also error, but it is due to previous error
        {'input': {'input': '0 5 0', '0->1': '1 5 1', '1->2': '2 5 4'}, 'output': {'2->3': '3 5 4'}},
        # call[3]: apply first skill 0->1, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'0->1': '1 5 1'}},
        # call[4]: apply second skill 1->2, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'1->2': '2 5 2'}},
        # call[5]: apply third skill 2->3, GT = 3 5 3
        {'input': {'input': '0 5 0', '0->1': '1 5 1', '1->2': '2 5 2'}, 'output': {'2->3': '3 5 3'}},
    ]
)
@patching(
    target_function=PatchedCalls.OPENAI_CHAT_COMPLETION.value,
    data=[
        {'input': {}, 'output': OpenaiChatCompletionMock(content='Transform 1 to 2')},
    ])
def test_agent_quickstart_three_skills_only_second_fail():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet, TransformSkill
    from adala.environments import StaticEnvironment

    agent = Agent(
        skills=LinearSkillSet(
            skills=[
                TransformSkill(name="0->1", instructions="...", input_template="Input: {input}", output_template="Output: {0->1}"),
                TransformSkill(name="1->2", instructions="...", input_template="Input: {0->1}", output_template="Output: {1->2}"),
                TransformSkill(name="2->3", instructions="...", input_template="Input: {1->2}", output_template="Output: {2->3}"),
            ]
        ),
        environment=StaticEnvironment(
            df=pd.DataFrame([
                ['0 5 0', '1 5 1', '2 5 2', '3 5 3'],
            ], columns=['input', 'gt_0', 'gt_1', 'gt_2']),
            ground_truth_columns={
                "0->1": "gt_0",
                "1->2": "gt_1",
                "2->3": "gt_2"
            }
        )
    )

    agent.learn(learning_iterations=1)

    # assert final instruction
    assert agent.skills['0->1'].instructions == '...'
    assert agent.skills['1->2'].instructions == 'Transform 1 to 2'
    assert agent.skills['2->3'].instructions == '...'
