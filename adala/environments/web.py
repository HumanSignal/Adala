import requests
import time
from typing import Optional
from .base import Environment
from .servers.base import GroundTruth
from adala.skills import SkillSet
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


class WebEnvironment(Environment):
    """
    Web environment interacts with server API to request feedback and retrieve ground truth.
    Following endpoints are expected:
    - POST /feedback
    - GET /ground-truth
    """
    url: str

    def request_feedback(self, skill_set: SkillSet, predictions: InternalDataFrame):
        requests.post(f'{self.url}/feedback', json={
            'skills': [dict(skill) for skill in skill_set.skills.values()],
            'predictions': predictions.reset_index().to_dict(orient='records')
        }, timeout=3)

    def get_gt_records(self):
        gt_records = requests.get(f'{self.url}/ground-truth', timeout=3).json()
        gt_records = [GroundTruth(**r) for r in gt_records]
        gt_records = [r for r in gt_records if r.gt_data or r.gt_match]
        return gt_records

    def get_ground_truth_dataset(self, wait: Optional[float] = None) -> InternalDataFrame:
        gt_records = []
        if wait:
            with Progress() as progress:
                task = progress.add_task(f"Waiting for ground truth {wait} seconds...", total=wait)
                while not gt_records and not progress.finished:
                    st = time.time()
                    gt_records = self.get_gt_records()
                    if not gt_records:
                        time.sleep(10)
                        time_elapsed = time.time() - st
                        progress.advance(task, time_elapsed)
        else:
            gt_records = self.get_gt_records()

        if not gt_records:
            raise RuntimeError('No ground truth found.')

        gt = defaultdict(dict)
        for g in gt_records:
            gt[g.skill_name][g.prediction_id] = g.gt_data or True

        df = InternalDataFrame({skill: InternalSeries(g) for skill, g in gt.items()})

        return df

