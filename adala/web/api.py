
import json

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from starlette.status import HTTP_201_CREATED

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi import APIRouter

from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED
)

from adala.web.routes import runtimes, agents, skills, versions, envs

router = APIRouter()

router.include_router(router=agents.router, tags=["Agents"], prefix="/api/agents")
router.include_router(router=skills.router, tags=["Skills"], prefix="/api/skills")
router.include_router(router=versions.router, tags=["Versions"], prefix="/api/skills")
router.include_router(router=runtimes.router, tags=["Runtimes"], prefix="/api/runtimes")
router.include_router(router=envs.router, tags=["Environments"], prefix="/api/envs")


# db = Database()
# db.connect()

# from .data_processing import DataProcessing


# @router.post("/upload/")
# async def upload_api(file: UploadFile = File(...)):
#     # with open(f"/temp/path/{file.filename}", "wb") as buffer:
#     #     buffer.write(file.file.read())
#     # data_processing = DataProcessing()
#     # return data_processing.send(f"/temp/path/{file.filename}")
#     pass

@router.get('/web')
async def index():
    # Serve the index.html file
    with open('static/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

