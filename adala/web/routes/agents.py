
from fastapi import APIRouter
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED
)

import adala.web.schema as schema

from ..models.agent import *
from ..repos.agents import *
from ..exceptions import AgentNotFoundException
from ..repos.skills import SkillsRepository, get_skills_repository
from ..repos.runtimes import get_runtimes_repository, RuntimesRepository


router = APIRouter()

## Agents

# @router.get("/agents/")
# async def list_agents_api():
#     # ... (similar structure to upload_api)
#     agents = AgentModel.select()
#     return [ a.serialize() for a in agents ]

@router.get("/", response_model=schema.Response[List[schema.Agent]])
def get_all_agents(
    skip: int = 0,
    limit: int = 100,
    agents_repo: AgentsRepository = Depends(get_agents_repository),    
    # current_user: User = Depends(get_current_user),
) -> schema.Response:
    """
    Retrieve all agents in the database within given limit and skip.

    Arguments:
    skip -- Amount of records to skip from the start (default 0)
    limit -- Maximum amount of records to retrieve (default 100)   
    """
    rts = agents_repo.get_all() # skip=skip, limit=limit)
    return schema.Response(data=rts)


@router.post("/", response_model=schema.Response[schema.Agent], status_code=HTTP_201_CREATED)
def create_agent(
    agent: schema.AgentCreate,
    agents_repo: AgentsRepository = Depends(get_agents_repository),
    # current_user: User = Depends(get_current_active_user),
) -> schema.Response:
    """
    Create new agent.
    """
    agent = agents_repo.create_with_objects(obj_create=agent)
    
    return schema.Response(
        data=dict(agent.__dict__,
                  runtimes=[ rt.id for rt in agent.runtimes ],
                  skills=[ sk.id for sk in agent.skills ],
                  envs=[ sk.id for sk in agent.envs ]),
        message="The agent was created successfully")


def get_current_agent(
    agent_id: str,
    repo: AgentsRepository = Depends(get_agents_repository),
    # current_user: User = Depends(get_current_user),
) -> Agent:
    """
    Check if agent with `agent_id` exists in database.
    """
    agent = repo.get(obj_id=agent_id)
    if not agent:
        raise AgentNotFoundException(
            message=f"Agent with id `{agent_id}` not found",
            status_code=HTTP_404_NOT_FOUND,
        )
    # if agent.owner_id != current_user.id:
    #     raise UserPermissionException(
    #         message="Not enough permissions", status_code=HTTP_403_FORBIDDEN
    #     )
    return agent


@router.get("/{agent_id}/", response_model=schema.Response[schema.Agent])
async def get_agent(agent: schema.Agent = Depends(get_current_agent)) -> schema.Response:
    """,
    Retrieve a agent by `agent_id`.
    """
    data = dict(
        agent.__dict__,
        runtimes=[ rt.id for rt in agent.runtimes ],
        skills=[ sk.id for sk in agent.skills ],
        envs=[ e.id for e in agent.envs ]
    )
    
    return schema.Response(data=data)


@router.post("/{agent_id}/skills/", response_model=schema.Response[schema.Skill], status_code=HTTP_201_CREATED)
async def get_agent(
        skill: schema.SkillCreate,
        agent: schema.Agent = Depends(get_current_agent),        
        skills_repo: SkillsRepository = Depends(get_skills_repository),
        agents_repo: AgentsRepository = Depends(get_agents_repository),
) -> schema.Response:
    """,
    Retrieve a agent by `agent_id`.
    """
    # TODO this is not the same transaction
    skill = skills_repo.create_with_first_version(obj_create=skill)
    agents_repo.add_skill_to_agent(agent, skill)
    
    return schema.Response(data=skill)


@router.get("/{agent_id}/skills/", response_model=schema.Response[List[schema.Skill]])
async def get_agent(agent: schema.Agent = Depends(get_current_agent)) -> schema.Response:
    """,
    Retrieve agent skills.
    """
    # import pdb
    # pdb.set_trace()
    
    return schema.Response(data=agent.skills)


# @router.get("/agetns/{agent_id}/skills/")
# async def get_agent_skills_api(agent_id: int):
#     agent = AgentModel.get_or_none(id=agent_id)
#     if not agent:
#         raise HTTPException(status_code=404, detail="Agent not found")
    
#     return agent.serialize_skills()


# @router.get("/agetns/{agent_id}/skills/")
# async def post_agent_skills_api(name: str):
#     agent = AgentModel.get_or_none(id=agent_id)
#     skill = SkillModel.create(agent=agent, name=name)
    
#     if not agent:
#         raise HTTPException(status_code=404, detail="Agent not found")
    
#     return skill.serialize()

import pandas as pd

from adala.agents import Agent
from adala.environments import StaticEnvironment
import adala.skills
# from adala.skills import ClassificationSkill
import adala.runtimes
import adala.skills.skillset
# from adala.runtimes import OpenAIChatRuntime
import json

from adala.web.connector import create_adala_instance 


@router.post("/{agent_id}/learn/")
async def learn_api(params: schema.RunLearnCreate,
                    agent: schema.Agent = Depends(get_current_agent),
                    runtimes_repo: RuntimesRepository = Depends(get_runtimes_repository),
                    repo: AgentsRepository = Depends(get_agents_repository)) -> schema.Response[schema.RunLearn]:
    """
    Execute a Learn job for the agent
    """
    adala_agent, skills_map, _, _ = create_adala_instance(agent)
    learn_run = repo.save_learn_run(params, agent)
    
    runtime = runtimes_repo.get_by(name=adala_agent.default_runtime)
    
    func = lambda **kwargs: repo.save_execution_log(learn_run=learn_run,
                                                    adala_agent=agent,
                                                    skills=skills_map,
                                                    runtime=runtime,
                                                    **kwargs)
    
    adala_agent.learn(learning_iterations=params.learning_iterations, accuracy_threshold=params.accuracy_threshold,
                      log_results_func=func)
    
    return schema.Response(data=learn_run)


@router.get("/{agent_id}/learn-status/")
async def learn_api( agent: schema.Agent = Depends(get_current_agent),
                    repo: AgentsRepository = Depends(get_agents_repository)) -> schema.Response:
    """
    Get the status of learn job.
    """
    data = repo.get_learn_status(agent)
    return schema.Response(data=data)


    # return agent.serialize()

# @router.post("/agetns/{agent_id}/predict/")
# async def predict_api():
#     agent = AgentModel.get_or_none(id=agent_id)
#     if not agent:
#         raise HTTPException(status_code=404, detail="Agent not found")

#     return agent.serialize()

# @router.post("/agetns/{agent_id}/deploy/")
# async def predict_api():
#     agent = AgentModel.get_or_none(id=agent_id)
#     if not agent:
#         raise HTTPException(status_code=404, detail="Agent not found")

#     return agent.serialize()


# @router.get("/agents/{agent_id}/metrics/")
# async def get_agent_metrics_api():
#     agent = AgentModel.get_or_none(id=agent_id)
#     if not agent:
#         raise HTTPException(status_code=404, detail="Agent not found")

#     return agent.serialize()

## Skills
