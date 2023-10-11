from pydantic import BaseModel
from typing import List
from .skills.base import Skill
from .memories.base import ShortTermMemory, LongTermMemory
from .tools.base import Tool


class Optimizer(BaseModel):
    """
    Base class for optimizers.
    """

    def optimize(
        self, skill: Skill,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        tools: List[Tool]
    ) -> Skill:
        """
        Optimize a skill and return the optimized skill.
        """
