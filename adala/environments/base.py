from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, Callable, Dict

from adala.utils.internal_data import InternalDataFrame, InternalSeries, InternalDataFrameConcat
from adala.utils.matching import fuzzy_match
from adala.skills.base import BaseSkill
from adala.skills.skillset import SkillSet
from adala.datasets import Dataset, DataFrameDataset


class GroundTruthSignal(BaseModel):
    """
    A model that represents the comparison between predictions and ground truth data,
    potentially holding information about matching results and errors per skill.

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
        errors (Optional[Dict[str, InternalDataFrame]]): A dictionary mapping skill names to DataFrames
            containing the errors between predictions and ground truth. Default is None.
            Each DataFrame has two columns ["predictions", "user-defined ground truth name"].
            User defined ground truth name is taken from Environment
            Example:
            ```json
                {
                    "skill_1": InternalDataFrame({
                        "predictions": ['a', 'b', 'c'],
                        "my_gr_truth": ['a', 'a', 'c']
                    }, index=[0, 1, 2])
                }
            ```
    """
    
    match: InternalDataFrame = Field(default_factory=InternalDataFrame)
    errors: Optional[Dict[str, InternalDataFrame]] = None

    def get_accuracy(self) -> InternalSeries:
        """
        Calculate the accuracy of predictions as the mean of matches.

        Returns:
            InternalSeries: A series representing the accuracy of predictions.

        Examples:
            

        """
        
        return self.match.mean()

    def get_errors(self, skill_name: str) -> InternalDataFrame:
        """
        Retrieve the errors associated with a particular skill.

        Args:
            skill_name (str): The name of the skill to retrieve errors for.

        Returns:
            InternalDataFrame: A DataFrame with two columns ["predictions", "ground_truth name"]
            representing the errors.

        Raises:
            AssertionError: If the error DataFrame does not have exactly two columns.
        """
        
        errors = self.errors[skill_name]
        assert len(errors.columns) == 2  # ["predictions", "ground_truth name"]
        return errors

    def __rich__(self):
        text = '[bold blue]Ground Truth Signal:[/bold blue]\n\n'
        text += f'\n[bold]Match[/bold]\n{self.match}'
        if self.errors is not None:
            for skill_name, errors in self.errors.items():
                text += f'\n[bold]Errors for {skill_name}[/bold]\n{errors}'
        return text

    class Config:
        arbitrary_types_allowed = True


class Environment(BaseModel, ABC):
    """
    An abstract base class that defines the structure and required methods for an environment
    in which machine learning models operate and are evaluated against ground truth data.

    Subclasses should implement methods to handle feedback requests, comparison to ground truth,
    dataset conversion, and state persistence.
    """
    data_stream: Optional[Dataset] = None
    ground_truth_columns: Optional[Dict[str, str]] = None
    matching_function: str = 'exact'
    matching_threshold: float = 0.8

    @field_validator('data_stream', mode='before')
    def _validate_data_stream(cls, v):
        """
        Validate the data stream field to ensure it is converted to DataFrameDataset if needed.

        Args:
            v: The value to validate.

        Returns:
            The validated value, possibly converted to DataFrameDataset.

        Raises:
            ValidationError: If the validation fails.
        """

        if isinstance(v, InternalDataFrame):
            return DataFrameDataset(df=v)
        return v

    def get_ground_truth_dataset(self, wait: Optional[float] = None) -> InternalDataFrame:
        """
        Get the ground truth dataset.

        Args:
            wait (Optional[float], optional): The timeout to wait for ground truth data to be available. Defaults to None.

        Returns:
            InternalDataFrame: The ground truth dataset.
        """
            
    @abstractmethod
    def request_feedback(self, skill_set: SkillSet, predictions: InternalDataFrame):
        """
        Abstract method to request user feedback on the predictions made by the model.

        Args:
            skill_set (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions made by the skills/models.
        """

    def compare_to_ground_truth(
        self,
        skill_set: SkillSet,
        predictions: InternalDataFrame,
        wait: Optional[float] = None,
    ) -> Optional[GroundTruthSignal]:
        """
        Compare the predictions with the ground truth using the specified matching function.

        Args:
            skill_set (SkillSet): The skill set being evaluated.
            predictions (InternalDataFrame): The predictions to compare with the ground truth.
            wait (Optional[float], optional): The timeout to wait for ground truth data to be available. Defaults to None.

        Returns:
            GroundTruthSignal: The resulting ground truth signal, with matches and errors detailed.

        Raises:
            NotImplementedError: If the matching_function is unknown.
        """

        ground_truth_match = InternalDataFrame()
        errors = {}
        ground_truth_dataset = self.get_ground_truth_dataset(wait=wait)
        if ground_truth_dataset.empty:
            return

        for skill_id, skill in skill_set.skills.items():
            if not self.ground_truth_columns or skill.name not in self.ground_truth_columns:
                gt_column = skill.name
            else:
                gt_column = self.ground_truth_columns[skill.name]
            gt = ground_truth_dataset[gt_column]
            pred = predictions[skill.name]
            # from ground truth dataset, select only the rows that are in the predictions
            gt, pred = gt.align(pred)
            nonnull_index = gt.notnull() & pred.notnull()
            gt = gt[nonnull_index]
            pred = pred[nonnull_index]
            # compare ground truth with predictions
            if self.matching_function == 'exact':
                gt_pred_match = gt == pred
            elif self.matching_function == 'fuzzy':
                gt_pred_match = fuzzy_match(gt, pred, threshold=self.matching_threshold)
            else:
                raise NotImplementedError(f'Unknown matching function {self.matching_function}')

            # for values with True, we assume them equal to predictions
            gt_pred_match[gt == True] = True
            gt_pred_match[gt == False] = False

            error_index = gt_pred_match[~gt_pred_match].index
            # concatenate errors - dataframe with two columns: predictions and ground truth
            errors[skill.name] = InternalDataFrameConcat([pred[error_index], gt[error_index]], axis=1)
            errors[skill.name].columns = ["predictions", gt_column]
            # concatenate matching columns
            ground_truth_match = InternalDataFrameConcat([
                # previous skills' ground truth matches
                ground_truth_match,
                # current skill's ground truth match
                gt_pred_match.rename(skill.name),
            ], axis=1)

        return GroundTruthSignal(
            match=ground_truth_match.reindex(predictions.index),
            errors=errors
        )

    def as_dataset(self) -> Dataset:
        """
        Abstract method to convert the environment's state into a dataset.

        Returns:
            Dataset: A dataset representing the environment's state.
        """
        return self.data_stream

    def save(self):
        """
        Save the current state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

        raise NotImplementedError

    def restore(self):
        """
        Restore the state of the BasicEnvironment.

        Raises:
            NotImplementedError: This method is not implemented for BasicEnvironment.
        """

        raise NotImplementedError

    class Config:
        arbitrary_types_allowed = True


class BasicEnvironment(Environment):
    """
    A concrete implementation of the Environment abstract base class,
    assuming the ground truth is provided and comparison is based on exact or fuzzy matching.

    Attributes:
        ground_truth_dataset (Union[InternalDataFrame, DataFrameDataset]): Dataset containing
            the ground truth data, defaulting to an empty DataFrameDataset.
        ground_truth_columns (Dict[str, str]): A dictionary mapping skill names to their corresponding
            ground truth columns in the dataset.
        matching_function (str): The name of the matching function to use, defaults to 'exact'.
        matching_threshold (float): The threshold for fuzzy matching, defaults to 0.8.
    """    
    ground_truth_dataset: DataFrameDataset = None

    @field_validator('ground_truth_dataset', mode='before')
    def _validate_ground_truth_dataset(cls, v):
        """
        Validate the ground_truth_dataset field to ensure it is converted to DataFrameDataset if needed.

        Args:
            v: The value to validate.

        Returns:
            The validated value, possibly converted to DataFrameDataset.

        Raises:
            ValidationError: If the validation fails.
        """
        
        if isinstance(v, InternalDataFrame):
            return DataFrameDataset(df=v)
        return v

    def request_feedback(self, skills: SkillSet, predictions: InternalDataFrame):
        """
        In the BasicEnvironment, this method is a placeholder as ground truth is already provided with the input data.

        Args:
            skill (BaseSkill): The skill being evaluated.
            predictions (InternalDataFrame): The predictions to be reviewed.
        """

    def get_ground_truth_dataset(self, wait: Optional[float] = None) -> InternalDataFrame:
        """
        Get the ground truth dataset.

        Returns:
            InternalDataFrame: The ground truth dataset.
        """
        return self.ground_truth_dataset.df

    def as_dataset(self) -> Dataset:
        """
        Return the dataset containing the ground truth data.

        Returns:
            Dataset: The ground truth dataset as a DataFrameDataset.
        """
        if self.ground_truth_dataset is not None:
            return self.ground_truth_dataset
        return super(BasicEnvironment, self).as_dataset()
