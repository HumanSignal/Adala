from enum import Enum


class Status(str, Enum):
    CANCELED = "Canceled"
    COMPLETED = "Completed"
    FAILED = "Failed"
    INPROGRESS = "InProgress"
    PENDING = "Pending"

    def __str__(self) -> str:
        return str(self.value)
