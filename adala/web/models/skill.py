
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


class SkillModel(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    agent_id = Column(Integer, ForeignKey('agentmodel.id'))
    agent = relationship("AgentModel", back_populates="skills")

    versions = relationship("SkillVersionModel", back_populates="skill")
    
    sk_class_name = Column(SQLEnum(SkillsType))
    name = Column(String, unique=True)
    description = Column(Text)
    instructions = Column(Text)
    input_template = Column(Text)
    output_template = Column(Text)    
    
    field_schema = Column(JSON)
    skill_params = Column(JSON)
    
    verbose = Column(Boolean, default=False)

    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
