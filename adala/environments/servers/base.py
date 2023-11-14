import asyncio

import json
from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, Request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import aiosqlite


STORAGE_DB = 'feedback.db'


class GroundTruth(BaseModel):
    prediction_id: int
    skill_output: str
    gt_match: Optional[bool] = None
    gt_data: Optional[str] = None


class Prediction(BaseModel):
    id: int
    input: Dict[str, Any]
    skill_name: str
    output: str


router = APIRouter()


# Base class for the API
class BaseAPI(FastAPI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_router(router)

    async def init_db(self):
        print(f'Initializing database {STORAGE_DB}...')
        async with aiosqlite.connect(STORAGE_DB) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS ground_truth (
                    prediction_id INTEGER NOT NULL,
                    skill_name TEXT NOT NULL,
                    gt_match BOOLEAN,
                    gt_data TEXT,
                    PRIMARY KEY (prediction_id, skill_name)
                )
            ''')
            await db.commit()

    async def request_feedback(
        self,
        predictions: List[Prediction],
        skills: List[Dict[str, Any]],
        db: aiosqlite.Connection
    ):
        raise NotImplementedError

    async def retrieve_ground_truth(self, db: aiosqlite.Connection):
        cursor = await db.execute('SELECT prediction_id, skill_name, gt_match, gt_data FROM ground_truth')
        rows = await cursor.fetchall()
        return [GroundTruth(prediction_id=row[0], skill_name=row[1], gt_match=row[2], gt_data=row[3]) for row in rows]

    async def store_ground_truths(self, ground_truths: List[GroundTruth], db: aiosqlite.Connection):
        await db.executemany('''
            INSERT OR REPLACE INTO ground_truth (prediction_id, skill_name, gt_match, gt_data)
            VALUES (?, ?, ?, ?)
        ''', [(gt.prediction_id, gt.skill_name, gt.gt_match, gt.gt_data) for gt in ground_truths])
        await db.commit()


# Dependency for managing the database connection lifecycle
async def get_db() -> aiosqlite.Connection:
    async with aiosqlite.connect(STORAGE_DB) as db:
        yield db


# Instantiate the base API
app = BaseAPI()


@router.post("/feedback")
async def create_feedback(
    request: Request,
    predictions: List[Dict[str, Any]],
    skills: List[Dict[str, Any]],
    db: aiosqlite.Connection = Depends(get_db)
):
    app = request.app
    await app.request_feedback(predictions, skills, db)
    return {"message": "Feedback received successfully"}


@router.get("/ground-truth", response_model=List[GroundTruth])
async def get_ground_truth(request: Request, db: aiosqlite.Connection = Depends(get_db)):
    app = request.app
    ground_truths = await app.retrieve_ground_truth(db)
    return ground_truths


@app.on_event("startup")
async def on_startup():
    await app.init_db()
