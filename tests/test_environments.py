import pytest

from adala.memories.base import ShortTermMemory
from adala.skills.base import BaseSkill
from adala.utils.internal_data import InternalDataFrame
from adala.environments.base import BasicEnvironment


class TestSkill(BaseSkill):
    def analyze(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass

    def improve(self, *args, **kwargs):
        pass


@pytest.fixture
def basic_env():
    ground_truth_data = InternalDataFrame({"ground_truth": [1, 0, 1, 1]})
    return BasicEnvironment(ground_truth_dataset=ground_truth_data, ground_truth_column='ground_truth')


@pytest.fixture
def short_term_memory():
    return ShortTermMemory(predictions=InternalDataFrame({"some_skill": [1, 0, 1, 0]}))


@pytest.fixture
def some_skill():
    return TestSkill(name='some_skill', input_data_field="text")


def test_compare_to_ground_truth(basic_env, short_term_memory, some_skill):
    experience = basic_env.compare_to_ground_truth(some_skill, short_term_memory)

    assert experience is not None
    assert "evaluations" in experience.model_dump()
    assert experience.ground_truth_column_name == 'ground_truth'
    assert experience.match_column_name == 'ground_truth__x__some_skill'

    expected_evaluations = InternalDataFrame({
        "some_skill": [1, 0, 1, 0],
        "ground_truth__x__some_skill": [True, True, True, False]
    })

    assert experience.evaluations.equals(expected_evaluations)