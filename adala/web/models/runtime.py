

import pdb

from datetime import datetime

from pydantic import BaseModel
from typing import Generator, Generic, List, Optional, Type, TypeVar, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, Table, DateTime, Float
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSON

from fastapi import Depends
from fastapi.encoders import jsonable_encoder

from ..enum import RuntimesType, SkillsType, SkillsGroupType, SkillVersionType, ExecutionLogState, LearnRunState
from ..schema import * 

from .base import Base
from .associations import agent_runtime_assoc_table


class RuntimeModel(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True)
    description = Column(Text)
    rt_class_name = Column(SQLEnum(RuntimesType))
    runtime_params = Column(JSON)
    
    agents = relationship("AgentModel", secondary=agent_runtime_assoc_table, back_populates="runtimes")
    execution_logs = relationship("ExecutionLogModel", back_populates="runtime")
    
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # agents_default = relationship("AgentModel", back_populates="default_runtime")
    # agents_default_teacher = relationship("AgentModel", back_populates="default_teacher_runtime")
