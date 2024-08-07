import pandas as pd
import pytest

from adala.skills import LinearSkillSet, TransformSkill  # type: ignore
from adala.utils.internal_data import InternalDataFrame  # type: ignore
from adala.environments import StaticEnvironment  # type: ignore

NaN = float("nan")


@pytest.mark.parametrize(
    "skillset, predictions, ground_truth, ground_truth_columns, expected_match, expected_feedback",
    [
        # test single skill, full ground truth signal
        (
            # skills
            LinearSkillSet(
                skills=[
                    TransformSkill(
                        name="some_skill",
                        input_template="Input: {text}",
                        output_template="Output: {skill_output}",
                        instructions="Convert all letters to '1'",
                    )
                ]
            ),
            # predictions
            InternalDataFrame(
                {"text": list("abcd"), "skill_output": ["1", "0", "1", "0"]}
            ),
            # ground truths
            InternalDataFrame({"my_ground_truth": ["1", "1", "1", "1"]}),
            # ground truth
            {"skill_output": "my_ground_truth"},
            # match
            InternalDataFrame({"skill_output": [True, False, True, False]}),
            # feedback
            InternalDataFrame(
                {
                    "skill_output": [
                        "Prediction is correct.",
                        'Prediction is incorrect. Correct answer: "1"',
                        "Prediction is correct.",
                        'Prediction is incorrect. Correct answer: "1"',
                    ]
                }
            ),
        ),
        # test two linear skills, partial ground truth signal
        (
            # skills
            LinearSkillSet(
                skills=[
                    TransformSkill(
                        name="skill_1",
                        input_template="Input: {text}",
                        output_template="Output: {skill_1}",
                        instructions="...",
                    ),
                    TransformSkill(
                        name="skill_2",
                        input_template="Input: {text}",
                        output_template="Output: {skill_2}",
                        instructions="...",
                    ),
                ]
            ),
            # predictions
            InternalDataFrame(
                {
                    "text": list("abcd"),
                    "skill_1": ["1", "0", "1", "0"],
                    "skill_2": ["1", "0", "0", "1"],
                },
                index=[11, 22, 33, 44],
            ),
            # ground truths
            InternalDataFrame(
                {
                    "gt_1": [NaN, "0", NaN, "1"],
                    "gt_2": ["1", "0", "1", NaN],
                },
                index=[11, 22, 33, 44],
            ),
            {"skill_1": "gt_1", "skill_2": "gt_2"},
            # expected match
            InternalDataFrame(
                {
                    "skill_1": [NaN, True, NaN, False],
                    "skill_2": [True, True, False, NaN],
                },
                index=[11, 22, 33, 44],
            ),
            # expected feedback
            InternalDataFrame(
                {
                    "skill_1": [
                        NaN,
                        "Prediction is correct.",
                        NaN,
                        'Prediction is incorrect. Correct answer: "1"',
                    ],
                    "skill_2": [
                        "Prediction is correct.",
                        "Prediction is correct.",
                        'Prediction is incorrect. Correct answer: "1"',
                        NaN,
                    ],
                },
                index=[11, 22, 33, 44],
            ),
        ),
        # test two linear skills, no ground truth signal for one skill, different size of dataframes
        (
            # skills
            LinearSkillSet(
                skills=[
                    TransformSkill(
                        name="skill_1",
                        input_template="Input: {text}",
                        output_template="Output: {skill_1}",
                        instructions="...",
                    ),
                    TransformSkill(
                        name="skill_2",
                        input_template="Input: {text}",
                        output_template="Output: {skill_2}",
                        instructions="...",
                    ),
                ]
            ),
            # predictions
            InternalDataFrame(
                {
                    "text": list("abcd"),
                    "skill_1": ["1", "0", "1", "0"],
                    "skill_2": ["1", "0", "0", "1"],
                },
                index=[11, 22, 33, 44],
            ),
            # ground truths
            InternalDataFrame(
                {
                    "gt_1": [NaN, NaN],
                    "gt_2": ["1", "0"],
                },
                index=[99, 44],
            ),
            {"skill_1": "gt_1", "skill_2": "gt_2"},
            # expected match
            InternalDataFrame(
                {"skill_1": [NaN, NaN, NaN, NaN], "skill_2": [NaN, NaN, NaN, False]},
                index=[11, 22, 33, 44],
            ),
            # expected feedback
            InternalDataFrame(
                {
                    "skill_1": [NaN, NaN, NaN, NaN],
                    "skill_2": [
                        NaN,
                        NaN,
                        NaN,
                        'Prediction is incorrect. Correct answer: "0"',
                    ],
                },
                index=[11, 22, 33, 44],
            ),
        ),
    ],
)
def test_basic_env_compare_to_ground_truth(
    skillset,
    predictions,
    ground_truth,
    ground_truth_columns,
    expected_match,
    expected_feedback,
):
    basic_env = StaticEnvironment(
        df=ground_truth, ground_truth_columns=ground_truth_columns
    )

    fb = basic_env.get_feedback(skillset, predictions)
    # TODO: we should check the index type and dtype, but it's not working for empty and NaN dataframes
    pd.testing.assert_frame_equal(
        expected_match, fb.match, check_index_type=False, check_dtype=False
    )

    pd.testing.assert_frame_equal(
        expected_feedback, fb.feedback, check_index_type=False, check_dtype=False
    )
