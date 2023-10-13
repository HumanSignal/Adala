from .base import Agent, AgentStep
from .skills.skillset import LinearSkillSet
from typing import List


class MultiStepPlanner(Agent):
    """
    Multi step planner gets a goal and creates a multi-step plan which skills to develop to achieve the goal.
    """
    skill_set: LinearSkillSet

    def step(self, learn=True) -> AgentStep:
        raise NotImplementedError
