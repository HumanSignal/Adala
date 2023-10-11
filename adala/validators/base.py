from pydantic import BaseModel
from .datasets.base import Dataset, MutableDataset


class Validator(BaseModel):
    """
    Base class for validators.
    """

    def validate(self, original_dataset: Dataset, predictions: MutableDataset) -> MutableDataset:
        """
        Validate a dataset with predictions and return new Dataset with validation results:
        - predictions
        - ground truth labels
        - errors
        """
