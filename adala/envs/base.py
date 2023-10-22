from pydantic import BaseModel

from adala.datasets.base import Dataset, BlankDataset, InternalDataFrame


class BaseEnvironment(BaseModel, ABC):
    """Environment contains the dataset that could be used by
    the agent to learn from.

    """
    
    dataset: None
    ground_truth_column: None
    
        
    def add_feedback(dataset):
        """
        """        

    def iterate():
        pass

    def batch_iterate(batch_size):
        pass


class StaticEnvironment(BaseEnvironment):
    """
    """
    pass


class DynamicEnvironment(BaseEnvironment):
    """
    """
    @abstractmethod
    def request_feedback(skill, dataset):
        """ Request external feedback
        """


class BlankEnvironment(StaticEnvironment):
    """
    """
    dataset: BlankDataset()
