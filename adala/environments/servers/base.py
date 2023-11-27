import asyncio

import json
from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, Request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import aiosqlite


STORAGE_DB = "feedback.db"


class Feedback(BaseModel):
    prediction_id: int
    prediction_column: str
    fb_match: Optional[bool] = None
    fb_message: Optional[str] = None


router = APIRouter()


# Base class for the API
class BaseAPI(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_router(router)

    async def init_db(self):
        print(f"Initializing database {STORAGE_DB}...")
        async with aiosqlite.connect(STORAGE_DB) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    prediction_id INTEGER NOT NULL,
                    prediction_column TEXT NOT NULL,
                    fb_match BOOLEAN,
                    fb_message TEXT,
                    PRIMARY KEY (prediction_id, prediction_column)
                )
            """
            )
            await db.commit()

    async def request_feedback(
        self,
        predictions: List[Dict[str, Any]],
        skills: List[Dict[str, Any]],
        db: aiosqlite.Connection,
    ):
        raise NotImplementedError

    async def retrieve_feedback(self, db: aiosqlite.Connection):
        cursor = await db.execute(
            "SELECT prediction_id, prediction_column, fb_match, fb_message FROM feedback"
        )
        rows = await cursor.fetchall()
        return [
            Feedback(
                prediction_id=row[0],
                prediction_column=row[1],
                fb_match=row[2],
                fb_message=row[3],
            )
            for row in rows
        ]

    async def store_feedback(self, feedbacks: List[Feedback], db: aiosqlite.Connection):
        await db.executemany(
            """
            INSERT OR REPLACE INTO feedback (prediction_id, prediction_column, fb_match, fb_message)
            VALUES (?, ?, ?, ?)
        """,
            [
                (fb.prediction_id, fb.prediction_column, fb.fb_match, fb.fb_message)
                for fb in feedbacks
            ],
        )
        await db.commit()


# Dependency for managing the database connection lifecycle
async def get_db() -> aiosqlite.Connection:
    async with aiosqlite.connect(STORAGE_DB) as db:
        yield db


# Instantiate the base API
app = BaseAPI()


@router.post("/request-feedback")
async def request_feedback(
    request: Request,
    predictions: List[Dict[str, Any]],
    skills: List[Dict[str, Any]],
    db: aiosqlite.Connection = Depends(get_db),
):
    app = request.app
    await app.request_feedback(predictions, skills, db)
    return {"message": "Feedback requested successfully"}


@router.get("/feedback", response_model=List[Feedback])
async def get_feedback(request: Request, db: aiosqlite.Connection = Depends(get_db)):
    app = request.app
    fb = await app.retrieve_feedback(db)
    return fb


@app.on_event("startup")
async def on_startup():
    await app.init_db()
