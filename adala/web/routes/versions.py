
""" """

from fastapi import APIRouter
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED
)

from ..models.version import *
from ..repos.versions import *
from ..exceptions import SkillVersionNotFoundException, ExecutionLogNotFoundException
from ..routes.skills import get_current_skill
from ..repos.agents import get_agents_repository, AgentsRepository
from ..repos.runtimes import get_runtimes_repository, RuntimesRepository

import adala.web.schema as schema
from ..connector import create_adala_instance
from ..settings import LOG_METRICS_RETURN_NUM

router = APIRouter()


def get_current_skill_version(
    skill_version_id: str,
    repo: SkillVersionsRepository = Depends(get_skill_versions_repository),
    # current_user: User = Depends(get_current_user),
) -> SkillVersion:
    """
    Check if skill_version with `skill_version_id` exists in database.
    """
    skill_version = repo.get(obj_id=skill_version_id)
    if not skill_version:
        raise SkillVersionNotFoundException(
            message=f"SkillVersion with id `{skill_version_id}` not found",
            status_code=HTTP_404_NOT_FOUND,
        )
    # if skill_version.owner_id != current_user.id:
    #     raise UserPermissionException(
    #         message="Not enough permissions", status_code=HTTP_403_FORBIDDEN
    #     )
    return skill_version


def get_current_execution_log(
    skill_version_id: str,
    repo: SkillVersionsRepository = Depends(get_skill_versions_repository),
    # current_user: User = Depends(get_current_user),
) -> SkillVersion:
    """
    Check if skill_version with `skill_version_id` exists in database.
    """
    skill_version = repo.get(obj_id=skill_version_id)
    execution_log = skill_version.execution_log
    if not execution_log:
        raise ExecutionLogNotFoundException(
            message=f"Learn log for Skill Version not found",
            status_code=HTTP_404_NOT_FOUND,
        )
    
    # if skill_version.owner_id != current_user.id:
    #     raise UserPermissionException(
    #         message="Not enough permissions", status_code=HTTP_403_FORBIDDEN
    #     )
    return execution_log


@router.get("/{skill_id}/versions/", response_model=schema.Response[List[schema.SkillVersion]])
def get_all_skill_versions(
    skip: int = 0,
    limit: int = 100,
    skill_versions_repo: SkillVersionsRepository = Depends(get_skill_versions_repository),    
    # current_user: User = Depends(get_current_user),
) -> schema.Response:
    """
    Retrieve all skill_versions.
    """
    rts = skill_versions_repo.get_all() # skip=skip, limit=limit)
    return schema.Response(data=rts)


@router.post("/{skill_id}/versions/", response_model=schema.Response[schema.SkillVersion], status_code=HTTP_201_CREATED)
def create_skill_version(        
        skill_version: schema.SkillVersionCreate,
        skill: schema.Skill = Depends(get_current_skill),
        skill_versions_repo: SkillVersionsRepository = Depends(get_skill_versions_repository),
        # current_user: User = Depends(get_current_active_user),
) -> schema.Response:
    """
    Create new skill_version.
    """
    skill_version = skill_versions_repo.create_with_defaults(skill=skill, obj_create=skill_version)
    
    return schema.Response(data=skill_version, message="The skill_version was created successfully")


@router.get("/{skill_id}/versions/{skill_version_id}/", response_model=schema.Response[schema.SkillVersion])
def get_skill_version(skill_version: schema.SkillVersion = Depends(get_current_skill_version)) -> schema.Response:
    """,
    Retrieve a skill_version by `skill_version_id`.
    """
    return schema.Response(data=skill_version)


### TODO
@router.put("/{skill_id}/versions/{skill_version_id}/",
            response_model=schema.Response[schema.SkillVersion])
def put_skill_version(
        skill_version_update: SkillVersionUpdate,
        skill_version: schema.SkillVersion = Depends(get_current_skill_version),
        skill_version_repo: SkillVersionsRepository = Depends(get_skill_versions_repository),
) -> schema.Response:
    """,
    Update a skill_version by `skill_version_id`.
    """
    updated = skill_version_repo.update(obj=skill_version, obj_update=skill_version_update)
    return schema.Response(data=updated, message="Version updated")


### TODO
@router.delete("/{skill_id}/versions/{skill_version_id}/",
            response_model=schema.Response[schema.SkillVersion])
def delete_skill_version(
        skill_version_update: SkillVersionUpdate,
        skill_version: schema.SkillVersion = Depends(get_current_skill_version),
        skill_version_repo: SkillVersionsRepository = Depends(get_skill_versions_repository),
) -> schema.Response:
    """,
    Delete specific skill.
    """
    updated = skill_version_repo.update(obj=skill_version, obj_update=skill_version_update)
    return schema.Response(data=updated, message="Version updated")


@router.get("/{skill_id}/versions/{skill_version_id}/logs/", response_model=schema.Response[List[schema.ExecutionLog]])
def get_skill_version_log(
        last: Optional[bool] = None,
        metrics: Optional[bool] = None,
        skill_version: schema.SkillVersion = Depends(get_current_skill_version),        
        skill_version_repo: SkillVersionsRepository = Depends(get_skill_versions_repository),
) -> schema.Response:
    """,
    Retrieve skill's version execution log.
    """
    idx = 1 if last else LOG_METRICS_RETURN_NUM
    logs = skill_version.execution_logs[0:idx]
    extended_logs = skill_version_repo.extend_logs_meta_data(logs)
    
    return schema.Response(data=extended_logs)


@router.post("/{skill_id}/versions/{skill_version_id}/execute/", response_model=schema.Response[schema.ExecutionLog])
def execute_skill_version(
        skill_instr: schema.SkillInstruction,
        skill: schema.Skill = Depends(get_current_skill),
        skill_version: schema.SkillVersion = Depends(get_current_skill_version),        
        agent_repo: AgentsRepository = Depends(get_agents_repository),
        runtimes_repo: RuntimesRepository = Depends(get_runtimes_repository),
        skill_version_repo: SkillVersionsRepository = Depends(get_skill_versions_repository)
) -> schema.Response:
    """
    Execute skill version, produces an execution log
    """
    # import pdb
    # pdb.set_trace()
    skill_version_repo.update_instruction(skill_version, skill_instr.instructions)
    adala_agent, skills_map, runtimes, env = create_adala_instance(skill.agent)
    
    predictions = adala_agent.skills.apply(
        input=env.df,
        runtime=runtimes[adala_agent.default_runtime],
        improved_skill=skill.name)

    runtime = runtimes_repo.get_by(name=adala_agent.default_runtime)
    
    _, execution_log = agent_repo.save_execution_log(learn_run=None, skill_version=skill_version, adala_agent=adala_agent, skills=skills_map,
                                                     learning_iterations=0, current_iteration=0, runtime=runtime,
                                                     inputs=env.df, predictions=predictions, feedback=None, train_skill_name=skill.name,
                                                     messages=None, new_instructions=skill_instr.instructions)    
    
    return schema.Response(data=execution_log)


@router.post("/{skill_id}/versions/{skill_version_id}/clone/", response_model=schema.Response[schema.SkillVersion])
def clone_skill_version(
        skill: schema.Skill = Depends(get_current_skill),
        skill_version: schema.SkillVersion = Depends(get_current_skill_version),        
        skill_version_repo: SkillVersionsRepository = Depends(get_skill_versions_repository)
) -> schema.Response:
    """,
    Clone skill version.
    """    
    new_version = skill_version_repo.clone_version(skill, skill_version)
    return schema.Response(data=new_version)


@router.post("/{skill_id}/versions/{skill_version_id}/set-active/", response_model=schema.Response[schema.SkillVersion])
def set_active_version(
        skill: schema.Skill = Depends(get_current_skill),
        skill_version: schema.SkillVersion = Depends(get_current_skill_version),        
        skill_version_repo: SkillVersionsRepository = Depends(get_skill_versions_repository)
) -> schema.Response:
    """
    Set skill version as active.
    """
    skill_version = skill_version_repo.set_active(skill, skill_version)
    return schema.Response(data=skill_version)


# @router.post("/{skill_id}/versions/{skill_version_id}/log/", response_model=schema.Response[schema.ExecutionLog])
# def get_skill_version(skill_version: schema.SkillVersion = Depends(get_current_skill_version)) -> schema.Response:
#     """,
#     Retrieve a skill_version by `skill_version_id`.
#     """
#     return schema.Response(data=skill_version.execution_log)


# @router.put("/{skill_id}/versions/{skill_version_id}/log/", response_model=schema.Response[schema.ExecutionLog])
# def get_skill_version(skill_version: schema.SkillVersion = Depends(get_current_skill_version)) -> schema.Response:
#     """,
#     Retrieve a skill_version by `skill_version_id`.
#     """
#     return schema.Response(data=skill_version.execution_log)

