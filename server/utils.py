# from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union
from pathlib import Path


class Settings(BaseSettings):
    """
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    """

    kafka_bootstrap_servers: Union[str, List[str]]

    model_config = SettingsConfigDict(
        # have to use an absolute path here so celery workers can find it
        env_file=(Path(__file__).parent / ".env"),
    )


def get_input_topic(job_id: str):
    return f"adala-input-{job_id}"


def get_output_topic(job_id: str):
    return f"adala-output-{job_id}"
