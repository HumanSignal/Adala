from .base import Agent, AgentStep, AgentArtifact
from adala.skills.base import Skill


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
            experience = self.skill.learn(self.dataset, runtime=self.runtime, memory=self.memory)
        else:
            self.skill.apply(self.dataset, runtime=self.runtime)
            experience = None

        if self.memory:
            self.memory.remember(experience)

        return AgentStep(
            artifact=AgentArtifact(
                experience=experience
            ),
            is_last=True
        )
