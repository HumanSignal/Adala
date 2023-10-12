from .base import Skill
from .datasets.base import Dataset, MutableDataset


class CodeSkill(Skill):
    """
    Code skill writes code to process input dataset and produce text output
    """

    def apply(self, dataset: Dataset) -> MutableDataset:
        pass