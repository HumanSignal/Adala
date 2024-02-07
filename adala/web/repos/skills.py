
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

from ..db import get_db
from .base import BaseRepository
from ..models.skill import SkillModel
from ..models.version import SkillVersionModel


class SkillsRepository(BaseRepository[SkillModel, SkillCreate, SkillUpdate]):
    """ """
    def create_with_first_version(self, obj_create):
        """ """
        params = obj_create.dict()
        obj = self.model(**params)
        
        version = SkillVersionModel(skill=obj, version_type=SkillVersionType.USER_DRAFT,
                                    instructions=obj.instructions)
        self.db.add(version)
        
        self.db.commit()
        self.db.refresh(obj)
        return obj



def get_skills_repository(session: Session = Depends(get_db)) -> SkillsRepository:
    return SkillsRepository(db=session, model=SkillModel)
