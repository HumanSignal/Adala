

import pdb

from datetime import datetime

from pydantic import BaseModel
from typing import Generator, Generic, List, Optional, Type, TypeVar, Any

from sqlalchemy import create_engine, update, desc, asc
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
from ..models.version import SkillVersionModel


class SkillVersionsRepository(BaseRepository[SkillVersionModel, SkillVersionCreate, SkillVersionUpdate]):
    """ """

    def set_active(self, skill, skill_version):
        """ """
        return skill_version
    
    
    def create_with_defaults(self, use="last", skill=None, obj_create=None):        
        """ """
        obj_create.instructions = skill.instructions
        version = self.create(obj_create=obj_create)
        
        return version
    
    def extend_logs_meta_data(self, logs):
        """ """
        # Calculate here any meta data we want to attach to the raw
        # log, things like metrics, inputs for the JS plots and alike
        # log["meta_data"] = {
        #     "hello": "world"
        # }
        
        return logs

    # def get_all(self) -> List[SkillVersionModel]:
    #     """
    #     Return all objects from specific db table.
    #     """
    #     return self.db.query(self.model).order_by(asc(self.model.created_date)).all()

    def clone_version(self, skill, skill_version):
        """ """
        new_version = SkillVersionModel(**{c.key: getattr(skill_version, c.key) for c in SkillVersionModel.__table__.columns if c.key not in ["id", "version_type", "accuracy", "execution_logs", "created_date", "updated_date", "learn_run", "learn_run_id"]},
                                   version_type=SkillVersionType.USER_DRAFT)

        skill.versions.append(new_version)
        
        self.db.add(new_version)
        self.db.commit()
        self.db.refresh(new_version)

        return new_version
    
    
    def update_instruction(self, version, instructions):
        """ """
        self.db.execute(update(SkillVersionModel)
                        .where(SkillVersionModel.id == version.id)
                        .values(instructions=instructions))
        self.db.commit()
        


def get_skill_versions_repository(session: Session = Depends(get_db)) -> SkillVersionsRepository:
    return SkillVersionsRepository(db=session, model=SkillVersionModel)
