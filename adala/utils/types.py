from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Union
from adala.utils.parse import parse_template


class BatchData(BaseModel):
    """
    Model for a batch of data submitted to a streaming job
    """

    job_id: str
    data: List[dict]


class ErrorResponseModel(BaseModel):
    message: str = Field(..., alias="_adala_message")
    details: str = Field(..., alias="_adala_details")

    model_config = ConfigDict(
        # omit other fields
        extra="ignore",
        # guard against name collisions with other fields
        populate_by_name=False,
    )
