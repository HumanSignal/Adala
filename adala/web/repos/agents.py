
import pdb
import json

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
from ..models.agent import AgentModel
from ..models.runtime import RuntimeModel
from ..models.skill import SkillModel
from ..models.env import EnvModel
from ..models.learnrun import LearnRunModel
from ..models.executionlog import ExecutionLogModel
from ..models.version import SkillVersionModel



class AgentsRepository(BaseRepository[AgentModel, AgentCreate, AgentUpdate]):
    """ """
    
    def get_learn_status(agent):
        """ """

        # TODO we need to return real status here        
        # we need to somehow pick the latest one we also need to check
        # if there is already one running and if it is then we need to
        # block it
        return {
            "status": "in progress...",
            "learning_iterations": 5,
            "current_iteration": 2
        }

    
    def save_learn_run(self, obj_create, agent):
        params = obj_create.dict()
        obj = LearnRunModel(**params, agent=agent)

        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)

        return obj


    def save_execution_log(self, learn_run=None, skill_version=None, adala_agent=None, skills=None,
                                learning_iterations=0, current_iteration=0, runtime=None,
                                inputs=None, predictions=None, feedback=None, train_skill_name=None,
                                messages=None, new_instructions=None):
        """ """
        if train_skill_name not in skills:
            raise Exception("Skill should be found in the hash")

        if not new_instructions:
            new_instructions="No new instructions"
        
        skill = skills.get(train_skill_name)
        
        COLUMN_NAME="sentiment"
        
        pred_array = predictions[COLUMN_NAME].to_json(orient='records')
        records_len = len(predictions)
        
        if feedback is not None:
            match_json = feedback.match[COLUMN_NAME].to_json(orient='records')
            accuracy = feedback.get_accuracy()
        else:
            match_array = predictions[COLUMN_NAME] == inputs[COLUMN_NAME]
            match_json = match_array.to_json(orient="records")
            
            accuracy = match_array.mean()

        
        if skill_version:
            log = ExecutionLogModel(skill_version=skill_version,
                                    runtime=runtime,
                                    current_iteration=current_iteration,
                                    accuracy=accuracy,
                                    records_len=records_len,
                                    predictions=pred_array,
                                    feedback_match=match_json)
            skill_version.accuracy = accuracy
            self.db.add(log)
        else:
            log = ExecutionLogModel(runtime=runtime,
                                    current_iteration=current_iteration,
                                    accuracy=accuracy,
                                    records_len=records_len,
                                    predictions=pred_array,
                                    feedback_match=match_json)

        
            # TODO: we will need to start saving environment version when env becomes versioned.
            # TODO: do we create a version of the skill even when the instruction is the same?
            skill_version = SkillVersionModel(skill_id=skill.id,
                                              execution_logs=[ log ],
                                              learn_run=learn_run,
                                              accuracy=accuracy,
                                              version_type=SkillVersionType.SYSTEM,
                                              instructions=new_instructions)

            self.db.add(skill_version)

        # update state
        if learn_run:
            learn_run.state = LearnRunState.RUNNING
            self.db.add(learn_run)
            
        if learn_run and learn_run.learning_iterations == current_iteration:
            learn_run.state = LearnRunState.DONE

        self.db.commit()
        self.db.refresh(skill_version)
        self.db.refresh(log)
        
        return skill_version, log

    # # TODO: we need to rename execution_log as execution, not "learn", because "learn" can mean many things
    # def save_execution_log(self, learn_run=None, skill_version=None, adala_agent=None, skills=None,
    #                    learning_iterations=None, current_iteration=None,
    #                    inputs=None, predictions=None, feedback=None, train_skill_name=None,
    #                    messages=None, new_instructions=None):        
    #     """ """

    #     # pdb.set_trace()

    #     if train_skill_name not in skills:
    #         raise Exception("Skill should be found in the hash")
            
    #     skill = skills.get(train_skill_name)
        
    #     if not new_instructions:
    #         new_instructions="No new instructions"

    #     # TODO how do we get the column names here?
    #     # pdb.set_trace()
    #     COLUMN_NAME="sentiment"

    #     # pdb.set_trace()
        
    #     pred_array = predictions[COLUMN_NAME].to_json(orient='records')
    #     match_array = feedback.match[COLUMN_NAME].to_json(orient='records')

    #     accuracy = feedback.get_accuracy()
        
    #     log = ExecutionLogModel(current_iteration=current_iteration,
    #                             accuracy=accuracy,
    #                             predictions=pred_array,
    #                             feedback_match=match_array)

        
    #     # TODO: we will need to start saving environment version when env becomes versioned.
    #     # TODO: do we create a version of the skill even when the instruction is the same?
    #     obj = SkillVersionModel(skill_id=skill.id,
    #                             execution_logs=[ log ],
    #                             learn_run=learn_run,
    #                             accuracy=accuracy,
    #                             version_type=SkillVersionType.SYSTEM,
    #                             instructions=new_instructions)

    #     # Learn Run state
    #     learn_run.state = LearnRunState.RUNNING
        
    #     if learn_run.learning_iterations == current_iteration:
    #         learn_run.state = LearnRunState.DONE

    #     self.db.add(learn_run)
    #     self.db.add(obj)
        
    #     self.db.commit()
    #     self.db.refresh(obj)

    #     return obj
        
    def create_with_objects(self, obj_create):
        """ """
        lookup = {}
        child_objs = (("runtimes", RuntimeModel),
                      ("envs", EnvModel),
                      ("skills", SkillModel))

        params = obj_create.dict()
        lookup = {name: params.pop(name, []) for name, _ in child_objs}
        
        instance = self.model(**params)

        for name, model_class in child_objs:
            if lookup[name] is not None:
                for idx in lookup[name]:
                    getattr(instance, name).append(self.db.query(model_class).get(idx))
        
        self.db.add(instance)
        self.db.commit()
        self.db.refresh(instance)
        
        return instance

    def add_skill_to_agent(self, agent, skill):
        agent.skills.append(skill)
        self.db.commit()
        self.db.refresh(skill)
        
        return skill
    

def get_agents_repository(session: Session = Depends(get_db)) -> AgentsRepository:
    return AgentsRepository(db=session, model=AgentModel)
