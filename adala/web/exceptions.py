
# from fastapi import HTTPException, status


class BaseInternalException(Exception):
    """
    Base error class for inherit all internal errors.
    """

    def __init__(self, message: str, status_code: int) -> None:
        self.message = message
        self.status_code = status_code


class AgentNotFoundException(BaseInternalException):
    """
    Exception raised when `agent_id` field from JSON body not found.
    """


class RuntimeNotFoundException(BaseInternalException):
    """
    Exception raised when `runtime_id` field from JSON body not found.
    """


class EnvNotFoundException(BaseInternalException):
    """
    Exception raised when `env_id` field from JSON body not found.
    """

    
class NotCSVFileException(BaseInternalException):
    """
    Exception raised when uploaded file is not CSV formatted.
    """

    
class GTColumnNotInCSVException(BaseInternalException):
    """
    Exception raised when uploaded file is not CSV formatted.
    """
    

class SkillNotFoundException(BaseInternalException):
    """
    Exception raised when `skill_id` field from JSON body not found.
    """


class SkillVersionNotFoundException(BaseInternalException):
    """
    Exception raised when `skill_id` field from JSON body not found.
    """


class ExecutionLogNotFoundException(BaseInternalException):
    """
    Exception raised when `skill_id` field from JSON body not found.
    """
