from rich import print
from rich.prompt import Prompt
from .base import StaticEnvironment
from adala.skills import SkillSet
from adala.utils.internal_data import InternalDataFrame
from adala.utils.logs import print_series
from adala.datasets import Dataset, DataFrameDataset


class ConsoleEnvironment(StaticEnvironment):

    def request_feedback(self, skill_set: SkillSet, predictions: InternalDataFrame):

        ground_truth_dataset = []
        for _, prediction in predictions.iterrows():
            print_series(prediction)
            pred_row = prediction.to_dict()
            for skill in skill_set.skills.values():
                ground_truth = Prompt.ask(
                    f'Does this prediction match "{skill.name}"? ("Yes" or provide your answer)', default="Yes")
                if ground_truth == "Yes":
                    pass
                else:
                    pred_row[skill.name] = ground_truth
            ground_truth_dataset.append(pred_row)

        self.df = InternalDataFrame(ground_truth_dataset)
