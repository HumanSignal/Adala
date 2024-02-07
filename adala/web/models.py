""" """
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

from .enum import RuntimesType, SkillsType, SkillsGroupType, SkillVersionType, ExecutionLogState, LearnRunState
from .schema import * 

from .models.base import Base
from .models.agent import AgentModel
from .models.executionlog import ExecutionLogModel
from .models.learnrun import LearnRunModel
from .models.runtime import  RuntimeModel
from .models.skill import SkillModel
from .models.version import SkillVersionModel

# from .db import get_db

# def init_models(): 
#     for m in [AgentModel, SkillModel, RuntimeModel, ExecutionLogModel, LearnRunModel, SkillVersionModel, association_table]:
#         # m.metadata.drop_all(engine)
#         # m.metadata.create_all(engine)













