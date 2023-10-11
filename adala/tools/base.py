from pydantic import BaseModel
from .datasets.base import Dataset, MutableDataset


class Tool(BaseModel):
    """
    Base class for tools.
    """

    def use(self, dataset: Dataset, predictions_and_ground_truth: MutableDataset) -> str:
        """
        Use a tool on a dataset and return a string with results.
        """
        pass
