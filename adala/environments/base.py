from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Optional, ClassVar
from adala.utils.internal_data import (
    InternalDataFrame,
    InternalSeries,
)
from adala.skills.skillset import SkillSet
from adala.utils.registry import BaseModelInRegistry


class EnvironmentFeedback(BaseModel):
    """
    A class that represents the feedback received from an environment,
    along with the calculated correctness of predictions.

    Attributes:
        match (InternalDataFrame): A DataFrame indicating the correctness of predictions.
                                   Each row corresponds to a prediction, and each column is a boolean indicating if skill matches ground truth.
                                   Columns are named after the skill names.
                                   Indices correspond to prediction indices.
                                   Example:
                                       ```
                                        | index | skill_1 | skill_2 | skill_3 |
                                        |-------|---------|---------|---------|
                                        | 0     | True    | True    | False   |
                                        | 1     | False   | False   | False   |
                                        | 2     | True    | True    | True    |
                                        ```
        feedback (InternalDataFrame): A DataFrame that contains ground truth feedback per each skill output
    """

    match: InternalDataFrame
    feedback: InternalDataFrame

    class Config:
        arbitrary_types_allowed = True

    def get_accuracy(self) -> InternalSeries:
        """
        Calculate the accuracy of predictions as the mean of matches.

        Returns:
            InternalSeries: A series representing the accuracy of predictions.
        """
        return self.match.mean()

    def __rich__(self):
        text = "[bold blue]Environment Feedback:[/bold blue]\n\n"
        text += f"\n[bold]Match[/bold]\n{self.match}"
        if self.feedback is not None:
            text += f"\n[bold]Feedback[/bold]\n{self.feedback}"
        return text


class Environment(BaseModelInRegistry):
    """
    An abstract base class that defines the structure and required methods for an environment
    in which machine learning models operate and are evaluated against ground truth data.

    Subclasses should implement methods to handle feedback requests, comparison to ground truth,
    dataset conversion, and state persistence.
    """

    @abstractmethod
    def initialize(self):
        """
        Initialize the environment, e.g by connecting to a database, reading file to memory or starting a stream.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    def finalize(self):
        """
        Finalize the environment, e.g by closing a database connection, writing memory to file or stopping a stream.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    def get_data_batch(self, batch_size: Optional[int]) -> InternalDataFrame:
        """
        Get a batch of data from data stream to be processed by the skill set.

        Args:
            batch_size (Optional[int], optional): The size of the batch. Defaults to None

        Returns:
            InternalDataFrame: The data batch.
        """

    @abstractmethod
    def get_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
    ) -> EnvironmentFeedback:
        """
        Request feedback for the predictions.

        Args:
            skills (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions to compare with the ground truth.
            num_feedbacks (Optional[int], optional): The number of feedbacks to request. Defaults to all predictions
        Returns:
            EnvironmentFeedback: The resulting ground truth signal, with matches and errors detailed.
        """

    @abstractmethod
    def save(self):
        """
        Save the current state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    def restore(self):
        """
        Restore the state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    class Config:
        arbitrary_types_allowed = True


class AsyncEnvironment(Environment, ABC):

    @abstractmethod
    async def initialize(self):
        """
        Initialize the environment, e.g by connecting to a database, reading file to memory or starting a stream.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    async def finalize(self):
        """
        Finalize the environment, e.g by closing a database connection, closing a file or stopping a stream.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    async def get_data_batch(self, batch_size: Optional[int]) -> InternalDataFrame:
        """
        Get a batch of data from data stream to be processed by the skill set.

        Args:
            batch_size (Optional[int], optional): The size of the batch. Defaults to None

        Returns:
            InternalDataFrame: The data batch.
        """

    @abstractmethod
    async def get_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
    ) -> EnvironmentFeedback:
        """
        Request feedback for the predictions.

        Args:
            skills (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions to compare with the ground truth.
            num_feedbacks (Optional[int], optional): The number of feedbacks to request. Defaults to all predictions
        Returns:
            EnvironmentFeedback: The resulting ground truth signal, with matches and errors detailed.
        """

    @abstractmethod
    async def set_predictions(self, predictions: InternalDataFrame):
        """
        Push predictions back to the environment.

        Args:
            predictions (InternalDataFrame): The predictions to push to the environment.
        """

    @abstractmethod
    async def save(self):
        """
        Save the current state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    @abstractmethod
    async def restore(self):
        """
        Restore the state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

    class Config:
        arbitrary_types_allowed = True
