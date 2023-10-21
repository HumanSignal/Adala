from .base import LongTermMemory, ShortTermMemory
from typing import Any


class FileMemory(LongTermMemory):

    filepath: str

    def remember(self, experience: ShortTermMemory):
        """
        Serialize experience in JSON and append to file
        """
        experience_json = experience.model_dump_json()
        with open(self.filepath, 'a') as f:
            f.write(experience_json + '\n')

    def retrieve(self, observations: ShortTermMemory) -> ShortTermMemory:
        """
        Retrieve experience from file
        """
        raise NotImplementedError
