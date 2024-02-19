import requests
import time
from typing import Optional
from .base import EnvironmentFeedback
from .static_env import StaticEnvironment
from .servers.base import Feedback
from adala.skills import SkillSet
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


class WebStaticEnvironment(StaticEnvironment):
    """
    Web environment interacts with server API to request feedback and retrieve ground truth.
    Following endpoints are expected:
    - POST /request-feedback
    - GET /feedback
    """

    url: str

    def _get_fb_records(self):
        fb_records = requests.get(f"{self.url}/feedback", timeout=3).json()
        fb_records = [Feedback(**r) for r in fb_records]
        return fb_records

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
        """

        predictions = predictions.sample(n=num_feedbacks)
        skills_payload = []
        for skill in skills.skills.values():
            skill_payload = dict(skill)
            skill_payload["outputs"] = skill.get_output_fields()
            skills_payload.append(skill_payload)

        payload = {
            "skills": skills_payload,
            "predictions": predictions.reset_index().to_dict(orient="records"),
        }

        requests.post(f"{self.url}/request-feedback", json=payload, timeout=3)

        # wait for feedback
        with Progress() as progress:
            task = progress.add_task(f"Waiting for feedback...", total=3600)
            fb_records = []
            while len(fb_records) < num_feedbacks:
                progress.advance(task, 10)
                time.sleep(10)
                fb_records = self._get_fb_records()

        if not fb_records:
            raise RuntimeError("No ground truth found.")

        match = defaultdict(list)
        feedback = defaultdict(list)
        index = []
        for f in fb_records:
            match[f.prediction_column].append(f.fb_match)
            feedback[f.prediction_column].append(f.fb_message)
            index.append(f.prediction_id)

        match = InternalDataFrame(match, index=index)
        feedback = InternalDataFrame(feedback, index=index)

        return EnvironmentFeedback(match=match, feedback=feedback)
