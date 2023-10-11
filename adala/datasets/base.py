from pydantic import BaseModel


class Dataset(BaseModel):
    """
    Base class for original datasets.
    """
    pass


class MutableDataset(Dataset):
    """
    Base class for mutable datasets: where agent workflows can add new data like predictions, ground truth, etc.
    They can be linked to original datasets via object references.
    """
    pass
