from .base import Skill
from adala.datasets.base import Dataset


class CodeSkill(Skill):
    """
    Code skill writes code to process input dataset and produce text output
    """

    def apply(self, dataset: Dataset) -> Dataset:
        pass