from .base import Agent, AgentStep
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
            experience = self.skill.learn(self.dataset, self.memory)
        else:
            self.skill.apply(self.dataset)
            experience = None

        self.memory.remember(experience)

        return AgentStep(artifacts=[experience], is_last=True)
