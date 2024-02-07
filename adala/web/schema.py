
from datetime import datetime
from typing import List, Optional, Dict, Any, TypeVar, Generic
from pydantic import BaseModel, Field
# from pydantic.generics import GenericModel
from .enum import RuntimesType, SkillsGroupType, EnvsType


class AgentBase(BaseModel):
    name: str
    description: Optional[str] = None

class AgentCreate(AgentBase):
    skills_group_class_name: SkillsGroupType
    runtimes: List[int]
    envs: Optional[List[int]] = None
    skills: Optional[List[int]] = None
    

class AgentUpdate(AgentBase):
    pass

class Agent(AgentBase):
    id: int
    runtimes: List[int]
    envs: List[int]
    skills: List[int]
    
    skills_group_class_name: str    
    
    default_runtime_id: Optional[int] = None
    default_teacher_runtime_id: Optional[int] = None

    
    created_date: datetime
    updated_date: datetime
    
    class Config:
        orm_mode = True


class SkillVersionBase(BaseModel):
    instructions: str    
    
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None


class SkillVersionCreate(SkillVersionBase):
    """ """
    version_type: Optional[str]


class SkillVersionUpdate(SkillVersionBase):
    pass


class SkillVersion(SkillVersionBase):
    id: int
    version_type: str
    accuracy: Optional[float] = None
    is_active: Optional[bool] = False
    
    class Config:
        orm_true = True

        
class SkillBase(BaseModel):
    name: str
    sk_class_name: str
    description: Optional[str] = None
    instructions: str
    input_template: str
    output_template: str
    field_schema: Optional[str] = None
    skill_params: Optional[str] = None    
    verbose: bool = False

    
class SkillCreate(SkillBase):
    pass

class SkillUpdate(SkillBase):
    pass

class Skill(SkillBase):
    id: int
    versions: List[SkillVersion]
    
    class Config:
        orm_mode = True


class ExecutionLogBase(BaseModel):
    current_iteration: int
    accuracy: float
    predictions: str
    feedback_match: str

    
class ExecutionLog(ExecutionLogBase):
    id: int
    meta_data: Optional[dict] = None
    created_date: datetime
    
    class Config:
        orm_mode = True

class ExecutionLogMetrics(BaseModel):
    id: int
    
    accuracy: float
    created_date: datetime
    
    class Config:
        orm_mode = True

        
class RuntimeBase(BaseModel):
    name: str
    description: Optional[str] = None
    rt_class_name: RuntimesType
    runtime_params: Optional[str] = None

class RuntimeCreate(RuntimeBase):
    pass

class RuntimeUpdate(RuntimeBase):
    pass

class Runtime(RuntimeBase):
    id: int
    ## agents: List[Agent] = []
    # execution_logs:
    execution_logs_count: Optional[int] = None
    
    created_date: datetime
    updated_date: datetime
    
    class Config:
        orm_mode = True

## Env

class EnvBase(BaseModel):
    name: str
    description: Optional[str] = None
    env_class_name: EnvsType
    env_params: Optional[str] = None
    num_items: Optional[int] = 0
    column_names: Optional[str] = None

class EnvCreate(EnvBase):
    """ """
    local_filename: Optional[str] = None
    orig_filename: Optional[str] = None

class EnvUpdate(EnvBase):
    pass

class Env(EnvBase):
    id: int
    ## agents: List[Agent] = []
    orig_filename: str
    
    created_date: datetime
    updated_date: datetime
    
    class Config:
        orm_mode = True


class RunLearnBase(BaseModel):
    learning_iterations: int = 1
    accuracy_threshold: float = 0.95


class RunLearnCreate(RunLearnBase):
    pass


class RunLearn(RunLearnBase):
    id: int

    class Config:
        orm_mode = True
        
    
ResponseData = TypeVar("ResponseData")

class Response(BaseModel, Generic[ResponseData]):
    success: bool = True
    data: Optional[ResponseData] = None
    message: Optional[str] = None
    errors: Optional[list] = None

    def dict(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore
        """Exclude `null` values from the response."""
        kwargs.pop("exclude_none", None)
        return super().dict(*args, exclude_none=True, **kwargs)


class SkillInstruction(BaseModel):
    instructions: str
