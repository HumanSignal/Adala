from pydantic import BaseModel
from abc import ABC, abstractmethod

from adala.datasets.base import Dataset


class Environment(BaseModel, ABC):
    """
    Base class for environments.
    """
    dataset: Dataset
