

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


class ExecutionLogModel(Base):
    """ """
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    skill_version_id = Column(Integer, ForeignKey('skillversionmodel.id'))
    skill_version = relationship("SkillVersionModel", back_populates="execution_logs")

    runtime_id = Column(Integer, ForeignKey('runtimemodel.id'))
    runtime = relationship("RuntimeModel", back_populates="execution_logs")
    
    current_iteration = Column(Integer)

    accuracy = Column(Float)
    # number of record it was executed on
    records_len = Column(Integer)
    
    predictions = Column(JSON)
    feedback_match = Column(JSON)

    state = Column(SQLEnum(ExecutionLogState))

    # this is to hold any error
    error_log = Column(Text)
    
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
