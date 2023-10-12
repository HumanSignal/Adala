from .base import Agent, AgentStep
from typing import Any
from .skills.base import Skill


class SingleShotAgent(Agent):
    """
    Single shot agent possesses a single skill and applies it immediately to a dataset.
    """
    skill: Skill

    def step(self, learn=True) -> AgentStep:
        """
        Run agent step and return results
        """
        if learn:
            self.skill.learn(self.dataset)
            result = None
        else:
            result = self.skill.apply(self.dataset)

        return AgentStep(artifacts=[result], is_last=True)
