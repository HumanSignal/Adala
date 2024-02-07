
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
from .associations import agent_runtime_assoc_table, agent_env_assoc_table


class AgentModel(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(Text)
    description = Column(Text)

    skills_group_class_name = Column(SQLEnum(SkillsGroupType))
    skills = relationship("SkillModel", back_populates="agent")

    learn_runs = relationship("LearnRunModel", back_populates="agent")
    
    runtimes = relationship("RuntimeModel", secondary=agent_runtime_assoc_table, back_populates="agents")
    envs = relationship("EnvModel", secondary=agent_env_assoc_table, back_populates="agents")

    # runtimes = relationship("RuntimeModel", secondary=association_table, back_populates="agents")    
    default_runtime_id = Column(Integer, ForeignKey('runtimemodel.id'))
    default_runtime = relationship("RuntimeModel", foreign_keys=[default_runtime_id])

    default_teacher_runtime_id = Column(Integer, ForeignKey('runtimemodel.id'))
    default_teacher_runtime = relationship("RuntimeModel", foreign_keys=[default_teacher_runtime_id])

    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
