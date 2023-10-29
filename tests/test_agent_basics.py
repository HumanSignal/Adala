import pandas as pd

from utils import patching, PatchedCalls


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
        # call[0]: apply first skill 0->1, first row
        {'input': {'input': '0 5 0'}, 'output': {'predictions': '1 5 1'}},
        # call[1]: apply first skill 0->1, second row
        {'input': {'input': '0 0 0'}, 'output': {'predictions': '1 5 1'}},
        # call[2]: analyze errors first skill 0->1
        {
            'input': {
                'input': '0 0 0',
                '0->1': '1 5 1',
                'gt_0': '1 1 1'
            },
            'output': {
                'reason': '0 transformed to 5 instead of 1'
            }
        },
        # call[3]: build error report for first skill 0->1
        {
            'input': {
                'predictions_and_errors': [{
                    'input': '0 0 0',
                    '0->1': '1 5 1',
                    'gt_0': '1 1 1',
                    'reason': '0 transformed to 5 instead of 1'
                }]},
            'output': '''\
                Input: 0 0 0
                Prediction: 1 5 1
                Ground Truth: 1 1 1
                Reason: 0 transformed to 5 instead of 1
            ''',
        },
        # call[4]: improve first skill 0->1
        {
            'input': {
                'error_analysis': '''\
                Input: 0 0 0
                Prediction: 1 5 1
                Ground Truth: 1 1 1
                Reason: 0 transformed to 5 instead of 1
            '''},
            'output': {
                'new_instruction': 'Transform 0 to 1'
            }
        },
        # call[5]: reapply skill 0->1, first row
        {'input': {'input': '0 5 0'}, 'output': {'predictions': '1 5 1'}},
        # call[6]: reapply skill 0->1, first row
        {'input': {'input': '0 0 0'}, 'output': {'predictions': '1 1 1'}},

    ]
)
def test_agent_quickstart_single_skill():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet
    from adala.environments import BasicEnvironment

    agent = Agent(
        skills=LinearSkillSet(
            skills={
                "0->1": "...",
            }
        ),
        environment=BasicEnvironment(
            ground_truth_dataset=pd.DataFrame([
                ['0 5 0', '1 5 1'],
                ['0 0 0', '1 1 1']
            ], columns=['input', 'gt_0']),
            ground_truth_columns={
                "0->1": "gt_0"
            }
        )
    )

    ground_truth_signal = agent.learn()

    # assert final instruction
    assert agent.skills['0->1'].instructions == 'Transform 0 to 1'
    # assert final accuracy for skill 0->1
    pd.testing.assert_series_equal(
        pd.Series({'0->1': 1.0}),
        ground_truth_signal.get_accuracy()
    )


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
        {'input': {'input': '0 5 0'}, 'output': {'predictions': '1 5 1'}},
        # call[1]: apply first skill 0->1, second row, GT = 1 1 1
        {'input': {'input': '0 0 0'}, 'output': {'predictions': '1 5 1'}},
        # call[2]: apply second skill 1->2, first row, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'predictions': '2 5 2'}},
        # call[3]: apply second skill 1->2, second row, GT = 2 2 2
        {'input': {'input': '0 0 0', '0->1': '1 5 1'}, 'output': {'predictions': '2 5 2'}},
        # call[4]: analyze errors first skill 0->1, error in the second row (0 0 0 -> 1 5 1)
        {
            'input': {
                'input': '0 0 0',
                '0->1': '1 5 1',
                'gt_0': '1 1 1'
            },
            'output': {
                'reason': '0 transformed to 5 instead of 1'
            }
        },
        # call[5]: build error report for first skill 0->1
        {
            'input': {
                'predictions_and_errors': [{
                    'input': '0 0 0',
                    '0->1': '1 5 1',
                    '1->2': '2 5 2',
                    'gt_0': '1 1 1',
                    'gt_1': '2 2 2',
                    'reason': '0 transformed to 5 instead of 1'
                }]},
            'output': '''\
                Input: 0 0 0
                Prediction: 1 5 1
                Ground Truth: 1 1 1
                Reason: 0 transformed to 5 instead of 1
            ''',
        },
        # call[6]: improve first skill 0->1
        {
            'input': {
                'error_analysis': '''\
                Input: 0 0 0
                Prediction: 1 5 1
                Ground Truth: 1 1 1
                Reason: 0 transformed to 5 instead of 1
            '''},
            'output': {
                'new_instruction': 'Transform 0 to 1'
            }
        },
        # call[7]: reapply first skill 0->1, first row, GT = 1 5 1
        {'input': {'input': '0 5 0'}, 'output': {'predictions': '1 5 1'}},
        # call[8]: reapply first skill 0->1, second row, GT = 1 1 1
        {'input': {'input': '0 0 0'}, 'output': {'predictions': '1 1 1'}},
        # call[9]: reapply second skill 1->2, first row, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'predictions': '2 2 2'}},
        # call[10]: reapply second skill 1->2, second row, GT = 2 2 2
        {'input': {'input': '0 0 0', '0->1': '1 1 1'}, 'output': {'predictions': '2 2 2'}},
        # call[11]: analyze errors second skill 1->2 (first row 2 2 2 instead of 2 5 2)
        {
            'input': {
                'input': '0 5 0',
                '0->1': '1 5 1',
                '1->2': '2 2 2',
                'gt_0': '1 5 1',
                'gt_1': '2 5 2'
            },
            'output': {
                'reason': '5 transformed to 2 instead of remaining 5'
            }
        },
        # call[12]: build error report for second skill 1->2
        {
            'input': {
                'predictions_and_errors': [{
                    'input': '0 5 0',
                    '0->1': '1 5 1',
                    '1->2': '2 2 2',
                    'gt_0': '1 5 1',
                    'gt_1': '2 5 2',
                    'reason': '5 transformed to 2 instead of remaining 5'
                }]},
            'output': '''\
                Input: 1 5 1
                Prediction: 2 2 2
                Ground Truth: 2 5 2
                Reason: 5 transformed to 2 instead of remaining 5
            ''',
        },
        # call[13]: improve second skill 1->2
        {
            'input': {
                'error_analysis': '''\
                Input: 1 5 1
                Prediction: 2 2 2
                Ground Truth: 2 5 2
                Reason: 5 transformed to 2 instead of remaining 5
            '''},
            'output': {
                'new_instruction': 'Transform 1 to 2'
            }
        },
        # call[14]: reapply second skill 1->2, first row, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'predictions': '2 5 2'}},
        # call[15]: reapply second skill 1->2, second row, GT = 2 2 2
        {'input': {'input': '0 0 0', '0->1': '1 1 1'}, 'output': {'predictions': '2 2 2'}},
    ]
)
def test_agent_quickstart_two_skills():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet
    from adala.environments import BasicEnvironment

    agent = Agent(
        skills=LinearSkillSet(
            skills={
                "0->1": "...",
                "1->2": "..."
            },
            skill_sequence=["0->1", "1->2"]
        ),
        environment=BasicEnvironment(
            ground_truth_dataset=pd.DataFrame([
                ['0 5 0', '1 5 1', '2 5 2'],
                ['0 0 0', '1 1 1', '2 2 2']
            ], columns=['input', 'gt_0', 'gt_1']),
            ground_truth_columns={
                "0->1": "gt_0",
                "1->2": "gt_1"
            }
        )
    )

    ground_truth_signal = agent.learn()

    # assert final instruction
    assert agent.skills['0->1'].instructions == 'Transform 0 to 1'
    assert agent.skills['1->2'].instructions == 'Transform 1 to 2'
    # assert final accuracy for skill 0->1
    pd.testing.assert_series_equal(
        pd.Series({'0->1': 1.0, '1->2': 1.0}),
        ground_truth_signal.get_accuracy()
    )


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
        {'input': {'input': '0 5 0'}, 'output': {'predictions': '1 5 1'}},
        # call[1]: apply second skill 1->2, GT = 2 5 2 -> ERROR!
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'predictions': '2 5 4'}},
        # call[3]: apply third skill 2->3, GT = 3 5 3 -> Also error, but it is due to previous error
        {'input': {'input': '0 5 0', '0->1': '1 5 1', '1->2': '2 5 4'}, 'output': {'predictions': '3 5 4'}},
        # call[4]: analyze errors for second skill 1->2 (2 5 4 instead of 2 5 2)
        {
            'input': {
                'input': '0 5 0',
                '0->1': '1 5 1',
                '1->2': '2 5 4',
                '2->3': '3 5 4',
                'gt_0': '1 5 1',
                'gt_1': '2 5 2',
                'gt_2': '3 5 3',
            },
            'output': {
                'reason': '1 transformed to 4 instead of 2'
            }
        },
        # call[5]: build error report for second skill 1->2
        {
            'input': {
                'predictions_and_errors': [{
                    'input': '0 5 0',
                    '0->1': '1 5 1',
                    '1->2': '2 5 4',
                    '2->3': '3 5 4',
                    'gt_0': '1 5 1',
                    'gt_1': '2 5 2',
                    'gt_2': '3 5 3',
                    'reason': '1 transformed to 4 instead of 2'
                }]},
            'output': '''\
                Input: 0 5 0
                Prediction: 2 5 4
                Ground Truth: 2 5 2
                Reason: 1 transformed to 4 instead of 2
            ''',
        },
        # call[6]: improve first skill 0->1
        {
            'input': {
                'error_analysis': '''\
                Input: 0 5 0
                Prediction: 2 5 4
                Ground Truth: 2 5 2
                Reason: 1 transformed to 4 instead of 2
            '''},
            'output': {
                'new_instruction': 'Transform 1 to 2'
            }
        },
        # call[7]: apply second skill 1->2, GT = 2 5 2
        {'input': {'input': '0 5 0', '0->1': '1 5 1'}, 'output': {'predictions': '2 5 2'}},
        # call[8]: apply third skill 2->3, GT = 3 5 3
        {'input': {'input': '0 5 0', '0->1': '1 5 1', '1->2': '2 5 2'}, 'output': {'predictions': '3 5 3'}},
    ]
)
def test_agent_quickstart_three_skills_only_second_fail():
    from adala.agents import Agent
    from adala.skills import LinearSkillSet
    from adala.environments import BasicEnvironment

    agent = Agent(
        skills=LinearSkillSet(
            skills={
                "0->1": "...",
                "1->2": "...",
                "2->3": "..."
            },
            skill_sequence=["0->1", "1->2", "2->3"]
        ),
        environment=BasicEnvironment(
            ground_truth_dataset=pd.DataFrame([
                ['0 5 0', '1 5 1', '2 5 2', '3 5 3'],
            ], columns=['input', 'gt_0', 'gt_1', 'gt_2']),
            ground_truth_columns={
                "0->1": "gt_0",
                "1->2": "gt_1",
                "2->3": "gt_2"
            }
        )
    )

    ground_truth_signal = agent.learn()

    # assert final instruction
    assert agent.skills['0->1'].instructions == '...'
    assert agent.skills['1->2'].instructions == 'Transform 1 to 2'
    assert agent.skills['2->3'].instructions == '...'
    # assert final accuracy for skill 0->1
    pd.testing.assert_series_equal(
        pd.Series({'0->1': 1.0, '1->2': 1.0, '2->3': 1.0}),
        ground_truth_signal.get_accuracy()
    )
