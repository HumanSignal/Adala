""" """
## Runtimes

from fastapi import APIRouter
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED
)

import adala.web.schema as schema

from ..models.runtime import *
from ..repos.runtimes import *
from ..exceptions import RuntimeNotFoundException

router = APIRouter()


def get_current_runtime(
    runtime_id: str,
    repo: RuntimesRepository = Depends(get_runtimes_repository),
    # current_user: User = Depends(get_current_user),
) -> Runtime:
    """
    Check if runtime with `runtime_id` exists in database.
    """
    runtime = repo.get(obj_id=runtime_id)
    if not runtime:
        raise RuntimeNotFoundException(
            message=f"Runtime with id `{runtime_id}` not found",
            status_code=HTTP_404_NOT_FOUND,
        )
    # if runtime.owner_id != current_user.id:
    #     raise UserPermissionException(
    #         message="Not enough permissions", status_code=HTTP_403_FORBIDDEN
    #     )
    return runtime


@router.get("/", response_model=schema.Response[List[schema.Runtime]])
def get_all_runtimes(
    skip: int = 0,
    limit: int = 100,
    runtimes_repo: RuntimesRepository = Depends(get_runtimes_repository),    
    # current_user: User = Depends(get_current_user),
) -> schema.Response:
    """
    Retrieve all runtimes.
    """
    rts = runtimes_repo.get_all_extended() # skip=skip, limit=limit)
    for rt in rts:
        rt.execution_logs_count = len(rt.execution_logs)
    
    return schema.Response(data=rts)

@router.post("/", response_model=schema.Response[schema.Runtime], status_code=HTTP_201_CREATED)
def create_runtime(
    runtime: schema.RuntimeCreate,
    runtimes_repo: RuntimesRepository = Depends(get_runtimes_repository),
    # current_user: User = Depends(get_current_active_user),
) -> schema.Response:
    """
    Create new runtime.
    """
    runtime = runtimes_repo.create(obj_create=runtime)
    return schema.Response(data=runtime, message="The runtime was created successfully")

@router.get("/{runtime_id}/", response_model=schema.Response[schema.Runtime])
def get_runtime(runtime: schema.Runtime = Depends(get_current_runtime)) -> schema.Response[schema.Runtime]:
    """,
    Retrieve a runtime by `runtime_id`.
    """
    return schema.Response(data=runtime)


@router.put("/{runtime_id}/", response_model=schema.Response[schema.Runtime])
def update_runtime(runtime: schema.Runtime = Depends(get_current_runtime)) -> schema.Response[schema.Runtime]:
    """,
    Update runtime details.
    """
    return schema.Response(data=runtime)

### TODO
@router.delete("/{runtime_id}/", response_model=schema.Response[schema.Runtime])
def delete_runtime(runtime: schema.Runtime = Depends(get_current_runtime)) -> schema.Response[schema.Runtime]:
    """,
    Update runtime details.
    """
    return schema.Response(data=runtime)

