import pandas as pd
import pytest

from adala.skills import LinearSkillSet, TransformSkill
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from adala.environments import StaticEnvironment

NaN = float("nan")


@pytest.mark.parametrize("skillset, predictions, ground_truth, ground_truth_columns, expected_match, expected_errors", [
    # test single skill, full ground truth signal
    (
        # skills
        LinearSkillSet(skills=[
            TransformSkill(
                name='some_skill',
                input_template="Input: {text}",
                output_template="Output: {skill_output}",
                instructions="Convert all letters to '1'"
            )
        ]),
        # predictions
        InternalDataFrame({"text": list('abcd'), "skill_output": ['1', '0', '1', '0']}),
        # ground truths
        InternalDataFrame({"my_ground_truth": ['1', '1', '1', '1']}),
        # ground truth
        {"skill_output": "my_ground_truth"},
        # match
        InternalDataFrame({"skill_output": [True, False, True, False]}),
        # errors
        {
            "skill_output": InternalSeries(data=['1', '1'], index=[1, 3], name="my_ground_truth")
        }
    ),
    # test two linear skills, partial ground truth signal
    (
        # skills
        LinearSkillSet(skills=[
            TransformSkill(name='skill_1', input_template="Input: {text}", output_template="Output: {skill_1}", instructions="..."),
            TransformSkill(name="skill_2", input_template="Input: {text}", output_template="Output: {skill_2}", instructions="...")
        ]),
        # predictions
        InternalDataFrame({
            "text": list('abcd'),
            "skill_1": ['1', '0', '1', '0'],
            "skill_2": ['1', '0', '0', '1']
        }, index=[11, 22, 33, 44]),
        # ground truths
        InternalDataFrame({
            "gt_1": [NaN, '0', NaN, '1'],
            "gt_2": ['1', '0', '1', NaN],
        }, index=[11, 22, 33, 44]),
        {"skill_1": "gt_1", "skill_2": "gt_2"},
        # expected match
        InternalDataFrame({
            "skill_1": [NaN, True, NaN, False],
            "skill_2": [True, True, False, NaN]
        }, index=[11, 22, 33, 44]),
        # expected errors
        {
            "skill_1": InternalSeries(data=['1'], index=[44], name="gt_1"),
            "skill_2": InternalSeries(data=['1'], index=[33], name="gt_2")
        }
    ),
    # test two linear skills, no ground truth signal for one skill, different size of dataframes
    (
        # skills
        LinearSkillSet(skills=[
            TransformSkill(name='skill_1', input_template="Input: {text}", output_template="Output: {skill_1}", instructions="..."),
            TransformSkill(name="skill_2", input_template="Input: {text}", output_template="Output: {skill_2}", instructions="...")
        ]),
        # predictions
        InternalDataFrame({
            "text": list('abcd'),
            "skill_1": ['1', '0', '1', '0'],
            "skill_2": ['1', '0', '0', '1']
        }, index=[11, 22, 33, 44]),
        # ground truths
        InternalDataFrame({
            "gt_1": [NaN, NaN],
            "gt_2": ['1', '0'],
        }, index=[99, 44]),
        {"skill_1": "gt_1", "skill_2": "gt_2"},
        # expected match
        InternalDataFrame({
            "skill_1": [NaN, NaN, NaN, NaN],
            "skill_2": [NaN, NaN, NaN, False]
        }, index=[11, 22, 33, 44]),
        # expected errors
        {
            "skill_1": InternalSeries(data=[], index=[], name="gt_1"),
            "skill_2": InternalSeries(data=['0'], index=[44], name="gt_2")
        }
    ),
])
def test_basic_env_compare_to_ground_truth(skillset, predictions, ground_truth, ground_truth_columns, expected_match, expected_errors):

    basic_env = StaticEnvironment(
        df=ground_truth,
        ground_truth_columns=ground_truth_columns
    )

    ground_truth_signal = basic_env.compare_to_ground_truth(skillset, predictions)

    # TODO: we should check the index type and dtype, but it's not working for empty and NaN dataframes
    pd.testing.assert_frame_equal(expected_match, ground_truth_signal.match, check_index_type=False, check_dtype=False), \
        f'Expected: {expected_match}\nGot: {ground_truth_signal.match}'

    if expected_errors is not None:
        for skill_output in skillset.get_skill_outputs():
            skill_errors = ground_truth_signal.errors[skill_output]
            expected_skill_errors = expected_errors[skill_output]
            pd.testing.assert_series_equal(expected_skill_errors, skill_errors, check_index_type=False, check_dtype=False), \
                f'Skill output {skill_output}\n\nExpected: {expected_skill_errors}\nGot: {skill_errors}'
