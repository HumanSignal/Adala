""" """

from fastapi import APIRouter
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED
)

import adala.web.schema as schema

from ..models.skill import *
from ..repos.skills import *
from ..exceptions import SkillNotFoundException

router = APIRouter()


def get_current_skill(
    skill_id: str,
    repo: SkillsRepository = Depends(get_skills_repository),
    # current_user: User = Depends(get_current_user),
) -> Skill:
    """
    Check if skill with `skill_id` exists in database.
    """
    skill = repo.get(obj_id=skill_id)
    if not skill:
        raise SkillNotFoundException(
            message=f"Skill with id `{skill_id}` not found",
            status_code=HTTP_404_NOT_FOUND,
        )
    # if skill.owner_id != current_user.id:
    #     raise UserPermissionException(
    #         message="Not enough permissions", status_code=HTTP_403_FORBIDDEN
    #     )
    return skill


@router.get("/", response_model=schema.Response[List[schema.Skill]])
def get_all_skills(
    skip: int = 0,
    limit: int = 100,
    skills_repo: SkillsRepository = Depends(get_skills_repository),    
    # current_user: User = Depends(get_current_user),
) -> schema.Response:
    """
    Retrieve all skills.
    """
    rts = skills_repo.get_all() # skip=skip, limit=limit)
    return schema.Response(data=rts)


@router.post("/", response_model=schema.Response[schema.Skill], status_code=HTTP_201_CREATED)
def create_skill(
    skill: schema.SkillCreate,
    skills_repo: SkillsRepository = Depends(get_skills_repository),
    # current_user: User = Depends(get_current_active_user),
) -> schema.Response:
    """
    Create new skill.
    """
    skill = skills_repo.create_with_first_version(obj_create=skill)
    
    return schema.Response(data=skill, message="The skill was created successfully")

### [TODO]
@router.put("/{skill_id}/", response_model=schema.Response[schema.Skill])
def update_skill(skill: schema.Skill = Depends(get_current_skill),
            skills_repo: SkillsRepository = Depends(get_skills_repository)) -> schema.Response:
    """,
    Update a skill by `skill_id`.
    """
    skills_repo.delete(skill.id)
    return schema.Response(data=skill)

@router.get("/{skill_id}/", response_model=schema.Response[schema.Skill])
def get_skill(skill: schema.Skill = Depends(get_current_skill)) -> schema.Response:
    """,
    Retrieve a skill by `skill_id`.
    """
    return schema.Response(data=skill)


@router.delete("/{skill_id}/", response_model=schema.Response[schema.Skill])
def delete_skill(skill: schema.Skill = Depends(get_current_skill),
            skills_repo: SkillsRepository = Depends(get_skills_repository)) -> schema.Response:
    """,
    Retrieve a skill by `skill_id`.
    """
    skills_repo.delete(skill.id)
    return schema.Response(data=skill)


# @router.get("/skills/")
# def list_skills_api():
#     # ... (Endpoint logic)
#     pass

# @router.post("/skills/")
# def post_skills_api():
#     # ... (Endpoint logic)
#     pass
