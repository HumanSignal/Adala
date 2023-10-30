import pandas as pd
import pytest

from adala.skills import LinearSkillSet, LLMSkill
from adala.utils.internal_data import InternalDataFrame
from adala.environments.base import BasicEnvironment

NaN = float("nan")


@pytest.mark.parametrize("skillset, predictions, ground_truth, ground_truth_columns, expected_match, expected_errors", [
    # test single skill, full ground truth signal
    (
        LinearSkillSet(skills=[LLMSkill(name='some_skill', input_data_field="text")]),
        InternalDataFrame({"text": list('abcd'), "some_skill": ['1', '0', '1', '0']}),
        InternalDataFrame({"my_ground_truth": ['1', '1', '1', '1']}),
        {"some_skill": "my_ground_truth"},
        # match
        InternalDataFrame({"some_skill": [True, False, True, False]}),
        # errors
        {
            "some_skill": InternalDataFrame({
                "predictions": ['0', '0'], "my_ground_truth": ['1', '1']}, index=[1, 3])
        }
    ),
    # test two linear skills, partial ground truth signal
    (
        # skills
        LinearSkillSet(skills=[
            LLMSkill(name='skill_1', input_data_field="text"),
            LLMSkill(name="skill_2", input_data_field="text")
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
            "skill_1": InternalDataFrame({
                "predictions": ['0'], "gt_1": ['1']}, index=[44]),
            "skill_2": InternalDataFrame({
                "predictions": ['0'], "gt_2": ['1']}, index=[33])
        }
    ),
    # test two linear skills, no ground truth signal for one skill, different size of dataframes
    (
        # skills
        LinearSkillSet(skills=[
            LLMSkill(name='skill_1', input_data_field="text"),
            LLMSkill(name="skill_2", input_data_field="text")
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
            "skill_1": InternalDataFrame({
                "predictions": [], "gt_1": []}, index=[]),
            "skill_2": InternalDataFrame({
                "predictions": ['1'], "gt_2": ['0']}, index=[44])
        }
    ),
])
def test_basic_env_compare_to_ground_truth(skillset, predictions, ground_truth, ground_truth_columns, expected_match, expected_errors):

    basic_env = BasicEnvironment(
        ground_truth_dataset=ground_truth,
        ground_truth_columns=ground_truth_columns
    )

    ground_truth_signal = basic_env.compare_to_ground_truth(skillset, predictions)

    # TODO: we should check the index type and dtype, but it's not working for empty and NaN dataframes
    pd.testing.assert_frame_equal(expected_match, ground_truth_signal.match, check_index_type=False, check_dtype=False), \
        f'Expected: {expected_match}\nGot: {ground_truth_signal.match}'

    if expected_errors is not None:
        for skill_name in skillset.skills:
            skill_errors = ground_truth_signal.errors[skill_name]
            expected_skill_errors = expected_errors[skill_name]
            pd.testing.assert_frame_equal(expected_skill_errors, skill_errors, check_index_type=False, check_dtype=False), \
                f'Skill {skill_name}\n\nExpected: {expected_skill_errors}\nGot: {skill_errors}'

