import logging

from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    """

    kafka_bootstrap_servers: Union[str, List[str]]

    model_config = SettingsConfigDict(
        env_file=".env",
    )


def dummy_handler(batch):
    """
    Dummy handler to test streaming output flow
    Can delete once we have a real handler
    """

    logger.info(f"\n\nHandler received batch: {batch}\n\n")


class ResultHandler(Enum):
    DUMMY = dummy_handler


def get_input_topic(job_id: str):
    return f"adala-input-{job_id}"


def get_output_topic(job_id: str):
    return f"adala-output-{job_id}"
