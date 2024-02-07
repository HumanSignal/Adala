
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


class SkillVersionModel(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    skill_id = Column(Integer, ForeignKey('skillmodel.id'))
    skill = relationship("SkillModel", back_populates="versions")

    learn_run_id = Column(Integer, ForeignKey('learnrunmodel.id'))
    learn_run = relationship("LearnRunModel", back_populates="skill_versions")

    accuracy = Column(Float, nullable=True)
    execution_logs = relationship("ExecutionLogModel", back_populates="skill_version", order_by="desc(ExecutionLogModel.created_date)")
    
    version_type = Column(SQLEnum(SkillVersionType), default=SkillVersionType.SYSTEM)
    is_active = Column(Boolean, default=False)
    instructions = Column(Text)
    
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
