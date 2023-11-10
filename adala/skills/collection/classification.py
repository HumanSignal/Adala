from adala.skills._base import TransformSkill
from typing import List


class ClassificationSkill(TransformSkill):
    """
    Classifies into one of the given labels.
    """
    labels: List[str]
