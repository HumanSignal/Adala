from .base import Skill
from .datasets.base import Dataset, MutableDataset


class LLMSkill(Skill):
    """
    LLM skill handles LLM to produce predictions given instructions
    """

    def apply(self, dataset: Dataset) -> MutableDataset:
        pass
