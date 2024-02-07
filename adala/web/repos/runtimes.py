

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
from sqlalchemy.orm import joinedload

from fastapi import Depends
from fastapi.encoders import jsonable_encoder

from ..enum import RuntimesType, SkillsType, SkillsGroupType, SkillVersionType, ExecutionLogState, LearnRunState
from ..schema import * 


from ..db import get_db
from .base import BaseRepository
from ..models.runtime import RuntimeModel


class RuntimesRepository(BaseRepository[RuntimeModel, RuntimeCreate, RuntimeUpdate]):
    """
    """
    
    def get_all_extended(self):
        """ """
        # TODO add .offset(skip).limit(limit)
        return self.db.query(self.model).options(joinedload(self.model.execution_logs)).all()
    
#         +    items = db.query(models.Item).options(joinedload(models.Item.children)).offset(skip).limit(limit).all()
#     return [ItemListResponse(
#         id=item.id, 
#         title=item.title, 
#         description=item.description, 
#         children_count=len(item.children)
#     ) for item in items]
# ```
#     pass


def get_runtimes_repository(session: Session = Depends(get_db)) -> RuntimesRepository:
    return RuntimesRepository(db=session, model=RuntimeModel)
