
""" """
## Envs

import io
import csv
import json
import pandas as pd

from typing import Union
from typing_extensions import Annotated

from fastapi import APIRouter, UploadFile, File, Body
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED
)

from uuid import uuid4
from pathlib import Path

import adala.web.schema as schema

from ..models.env import *
from ..repos.envs import *
from ..exceptions import EnvNotFoundException, NotCSVFileException, GTColumnNotInCSVException

router = APIRouter()

from ..settings import UPLOAD_PATH


def get_current_env(
    env_id: str,
    repo: EnvsRepository = Depends(get_envs_repository),
    # current_user: User = Depends(get_current_user),
) -> Env:
    """
    Check if env with `env_id` exists in database.
    """
    env = repo.get(obj_id=env_id)
    if not env:
        raise EnvNotFoundException(
            message=f"Env with id `{env_id}` not found",
            status_code=HTTP_404_NOT_FOUND,
        )
    # if env.owner_id != current_user.id:
    #     raise UserPermissionException(
    #         message="Not enough permissions", status_code=HTTP_403_FORBIDDEN
    #     )
    return env


@router.get("/", response_model=schema.Response[List[schema.Env]])
def get_all_envs(
    skip: int = 0,
    limit: int = 100,
    envs_repo: EnvsRepository = Depends(get_envs_repository),    
    # current_user: User = Depends(get_current_user),
) -> schema.Response:
    """
    Retrieve all envs.
    """
    rts = envs_repo.get_all() # skip=skip, limit=limit)
    return schema.Response(data=rts)


@router.post("/", response_model=schema.Response[schema.Env], status_code=HTTP_201_CREATED)
async def create_env(
        name: str = Body(...),
        description: Optional[str] = Body(None),
        env_class_name: EnvsType = Body(...),        
        env_params: Optional[str] = Body(None),
        
        # env: schema.EnvCreate = Body(...),
        file: Union[UploadFile, None] = None,
        envs_repo: EnvsRepository = Depends(get_envs_repository),
        
        # file: UploadFile = File(...)       
    # current_user: User = Depends(get_current_active_user),
) -> schema.Response:
    """
    Create new env.
    """
    env_p = EnvCreate(name=name, description=description,
                      env_class_name=env_class_name, env_params=env_params)
    
    if file:
        unique_filename = str(uuid4())

        # TODO this is hardcoded
        file_path = Path(UPLOAD_PATH) / unique_filename
        data = await file.read()
        
        try:
            df = pd.read_csv(io.StringIO(data.decode()))
            # reader = csv.DictReader(data.decode().splitlines())
            
            env_p.num_items = len(df)
            env_p.column_names = json.dumps(df.columns.tolist())
            
            # TODO column name is hardcoded for now
            # if "ground_truth" not in reader.fieldnames:
            #     raise GTColumnNotInCSVException(message=f"File has no groun truth column present",
            #                                     status_code=HTTP_404_NOT_FOUND)

            with file_path.open("wb") as buffer:
                buffer.write(data)

        except pd.errors.ParserError:
           raise NotCSVFileException(message=f"File is not CSV",
                                     status_code=HTTP_404_NOT_FOUND)

        # with file_path.open("wb") as buffer:
        #     buffer.write(await file.read())

        env_p.local_filename = unique_filename
        env_p.orig_filename = file.filename
           
    env = envs_repo.create(obj_create=env_p)
    return schema.Response(data=env, message="The env was created successfully")


@router.get("/{env_id}/", response_model=schema.Response[schema.Env])
def get_env(env: schema.Env = Depends(get_current_env)) -> schema.Response[schema.Env]:
    """,
    Retrieve a env by `env_id`.
    """
    return schema.Response(data=env)

### TODO
@router.put("/{env_id}/", response_model=schema.Response[schema.Env])
def update_env(env: schema.Env = Depends(get_current_env)) -> schema.Response[schema.Env]:
    """,
    Update env
    """
    return schema.Response(data=env)

### TODO
@router.delete("/{env_id}/", response_model=schema.Response[schema.Env])
def delete_env(env: schema.Env = Depends(get_current_env)) -> schema.Response[schema.Env]:
    """,
    Delete env
    """
    return schema.Response(data=env)


@router.get("/{env_id}/data/", response_model=schema.Response[List[Dict]])
def get_env_data(env: schema.Env = Depends(get_current_env)) -> schema.Response[List[Dict]]:
    """,
    Retrieve env data by `env_id`.
    """
    try:
        file_path = Path(UPLOAD_PATH) / env.local_filename
        data = pd.read_csv(file_path)
        
        return schema.Response(data=data.to_dict('records'))
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
