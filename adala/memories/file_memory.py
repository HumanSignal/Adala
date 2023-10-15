from .base import Memory
from adala.skills.base import Experience
from typing import Any


class FileMemory(Memory):

    filepath: str

    def remember(self, experience: Experience):
        """
        Serialize experience in JSON and append to file
        """
        experience_json = experience.json()
        with open(self.filepath, 'a') as f:
            f.write(experience_json + '\n')

    def retrieve(self, observations: Any) -> Experience:
        """
        Retrieve experience from file
        """
        raise NotImplementedError
