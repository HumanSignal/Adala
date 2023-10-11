from pydantic import BaseModel
from .datasets.base import Dataset, MutableDataset


class Observations(BaseModel):
    """
    Base class for observations.
    """
    pass


class Analyzer(BaseModel):
    """
    Base class for analyzers.
    """

    def analyze(self, predictions_and_ground_truth: MutableDataset) -> Observations:
        """
        Analyze a dataset with predictions and ground truth and return observations.
        """
        pass
