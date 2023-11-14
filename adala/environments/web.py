import requests
import time
from typing import Optional
from .base import StaticEnvironment
from .servers.base import GroundTruth
from adala.skills import SkillSet
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


class WebStaticEnvironment(StaticEnvironment):
    """
    Web environment interacts with server API to request feedback and retrieve ground truth.
    Following endpoints are expected:
    - POST /feedback
    - GET /ground-truth
    """
    url: str

    def get_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
    ):
        """
        Request feedback for the predictions.

        Args:
            skills (SkillSet): The set of skills/models whose predictions are being evaluated.
            predictions (InternalDataFrame): The predictions to compare with the ground truth.
            num_feedbacks (Optional[int], optional): The number of feedbacks to request. Defaults to all predictions
        """

        predictions = predictions.sample(n=num_feedbacks)
        skills_payload = []
        for skill in skills.skills.values():
            skill_payload = dict(skill)
            skill_payload['outputs'] = skill.get_output_fields()
            skills_payload.append(skill_payload)

        payload = {
            'skills': skills_payload,
            'predictions': predictions.reset_index().to_dict(orient='records')
        }

        requests.post(f'{self.url}/feedback', json=payload, timeout=3)

        # wait for feedback
        with Progress() as progress:
            task = progress.add_task(f"Waiting for feedback...", total=3600)
            gt_records = []
            while len(gt_records) < num_feedbacks:
                progress.advance(task, 10)
                time.sleep(10)
                gt_records = self.get_gt_records()

        if not gt_records:
            raise RuntimeError('No ground truth found.')

        gt = defaultdict(dict)
        for g in gt_records:
            gt[g.skill_output][g.prediction_id] = g.gt_data or True

        df = InternalDataFrame({skill: InternalSeries(g) for skill, g in gt.items()})

        return df

    def get_gt_records(self):
        gt_records = requests.get(f'{self.url}/ground-truth', timeout=3).json()
        gt_records = [GroundTruth(**r) for r in gt_records]
        gt_records = [r for r in gt_records if r.gt_data or r.gt_match]
        return gt_records
