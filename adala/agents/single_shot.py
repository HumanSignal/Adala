from .base import Agent, AgentStep
from adala.skills.base import BaseSkill


class SingleShotAgent(Agent):
    """
    Single shot agent possesses a single skill and applies it immediately to a dataset.
    """
    skill: BaseSkill = None

    def greet(self):
        greeting = f'''
Hi, I am a single shot agent.
I can use a single skill to process a dataset.
For example, you can instruct me to label your dataset.'''
        if self.skill:
            greeting += f'''
I have a skill {self.skill} which can label your dataset.'''
        else:
            greeting += f'''

I have no skills yet. To begin, you need to teach me a skill.
For example, try:

agent.skill = adala.LabelingSkill(labels=['positive', 'negative'])'''
        print(greeting)

    def run(self):
        if not self.skill:
            raise ValueError("Skill is not set")

        experience = self.skill.apply(dataset=self.dataset, runtime=self.runtime)
        return AgentStep(experience=experience, is_last=True)

    def learn(self):
        experience = self.skill.learn(self.dataset, runtime=self.runtime, memory=self.memory)
        if self.memory:
            self.memory.remember(experience)

        return AgentStep(experience=experience, is_last=True)
