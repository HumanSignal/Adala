import logging
import json

# from enum import Enum
from abc import abstractmethod
from pydantic import BaseModel, computed_field, ConfigDict, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

from adala.utils.registry import BaseModelInRegistry

from label_studio_sdk import Client


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


class ResultHandler(BaseModelInRegistry):

    @abstractmethod
    def __call__(self, batch):
        '''
        Callable to do something with a batch of results.
        '''

class DummyHandler(ResultHandler):
    """
    Dummy handler to test streaming output flow
    Can delete once we have a real handler
    """

    def __call__(self, batch):
        logger.info(f"\n\nHandler received batch: {batch}\n\n")


class LSEHandler(ResultHandler):
    """
    Handler to use the Label Studio SDK to load a batch of results back into a Label Studio project
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    api_key: str
    url: str
    job_id: str

    @computed_field
    def client(self) -> Client:
        return Client(
            api_key=self.api_key,
            url=self.url,
        )

    @model_validator(mode="after")
    def ready(self):
        # Need this to make POST requests using the SDK client
        self.client.headers.update(
            {
                'accept': 'application/json',
                'Content-Type': 'application/json',
            }
        )

        self.client.check_connection()

        return self

    def __call__(self, batch):
        logger.info(f"\n\nHandler received batch: {batch}\n\n")
        self.client.make_request(
            "POST",
            "/api/model-run/batch-predictions",
            data=json.dumps(
                {
                    'job_id': self.job_id,
                    'results': batch,
                }
            ),
        )


# class ResultHandler(Enum):
    # DUMMY = dummy_handler
    # LSE = LSEHandler


def get_input_topic(job_id: str):
    return f"adala-input-{job_id}"


def get_output_topic(job_id: str):
    return f"adala-output-{job_id}"
