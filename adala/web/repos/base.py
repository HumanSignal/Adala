
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

from ..models.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base repository with basic methods.
    """

    def __init__(self, db: Session, model: Type[ModelType]) -> None:
        """
        CRUD object with default methods to Create, Read, Update, Delete (CRUD).

        :param db: A SQLAlchemy Session object.
        :param model: A SQLAlchemy model class.
        """
        self.db = db
        self.model = model

    def get_all(self) -> List[ModelType]:
        """
        Return all objects from specific db table.
        """
        return self.db.query(self.model).all()

    def get(self, obj_id: str) -> Optional[ModelType]:
        """
        Get object by `id` field.
        """
        obj = self.db.query(self.model).filter(self.model.id == obj_id).first()

        # import pdb
        # pdb.set_trace()
        
        return obj

    def get_by(self, **kwargs) -> Optional[ModelType]:
        """
        Get object by `id` field.
        """
        obj = self.db.query(self.model).filter_by(**kwargs).first()        
        
        # import pdb
        # pdb.set_trace()
        
        return obj

    def create(self, obj_create: CreateSchemaType) -> ModelType:
        """
        Create new object in db table.
        """
        print(obj_create.dict())
        
        obj = self.model(**obj_create.dict())
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def update(self, obj: ModelType, obj_update: UpdateSchemaType) -> ModelType:
        """
        Update model object by fields from `obj_update` schema.
        """
        obj_data = jsonable_encoder(obj)
        update_data = obj_update.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(obj, field, update_data[field])
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def delete(self, obj_id: int) -> Optional[ModelType]:
        """
        Delete object.
        """
        obj = self.db.query(self.model).get(obj_id)
        self.db.delete(obj)
        self.db.commit()
        return obj
