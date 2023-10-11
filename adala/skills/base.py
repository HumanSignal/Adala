from pydantic import BaseModel
from .datasets.base import Dataset, MutableDataset


class Skill(BaseModel):
    """
    Base class for skills.
    Docstring serves as skill instructions.
    """

    def apply(self, dataset: Dataset) -> MutableDataset:
        """
        Apply a skill to a dataset and return dataset with predictions.
        """
