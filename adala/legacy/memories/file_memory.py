import json
from .base import Memory
from typing import Any


class FileMemory(Memory):
    filepath: str

    def remember(self, observation: str, experience: Any):
        """
        Serialize experience in JSON and append to file
        """
        with open(self.filepath) as f:
            memory = json.load(f)
        memory[observation] = experience
        with open(self.filepath, "w") as f:
            json.dump(memory, f, indent=2)

    def retrieve(self, observation: str) -> Any:
        """
        Retrieve experience from file
        """
        with open(self.filepath) as f:
            memory = json.load(f)
        return memory[observation]
