from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union
import logging
import os
from pathlib import Path
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


class Settings(BaseSettings):
    """
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    """

    kafka_bootstrap_servers: Union[str, List[str]] = "localhost:9093"
    kafka_retention_ms: int = 180000  # 30 minutes
    kafka_input_consumer_timeout_ms: int = 1500  # 1.5 seconds
    kafka_output_consumer_timeout_ms: int = 1500  # 1.5 seconds
    task_time_limit_sec: int = 60 * 60 * 6  # 6 hours

    model_config = SettingsConfigDict(
        # have to use an absolute path here so celery workers can find it
        env_file=(Path(__file__).parent / ".env"),
    )


def get_input_topic_name(job_id: str):
    topic_name = f"adala-input-{job_id}"

    return topic_name


def get_output_topic_name(job_id: str):
    topic_name = f"adala-output-{job_id}"

    return topic_name


def ensure_topic(topic_name: str):
    settings = Settings()
    bootstrap_servers = settings.kafka_bootstrap_servers
    retention_ms = settings.kafka_retention_ms

    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        client_id="topic_creator",
        api_version=(2, 5, 0),
    )

    topic = NewTopic(
        name=topic_name,
        num_partitions=1,
        replication_factor=1,
        topic_configs={"retention.ms": str(retention_ms)},
    )

    try:
        admin_client.create_topics(new_topics=[topic])
    except TopicAlreadyExistsError:
        # we shouldn't hit this case when KAFKA_CFG_AUTO_CREATE_TOPICS=false unless there is a legitimate name collision, so should raise here after testing
        pass


def delete_topic(topic_name: str):
    settings = Settings()
    bootstrap_servers = settings.kafka_bootstrap_servers

    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        client_id="topic_deleter",
        api_version=(2, 5, 0),
    )

    admin_client.delete_topics(topics=[topic_name])


def init_logger(name, level=LOG_LEVEL):
    """Set up a logger that respects the LOG_LEVEL env var

    Args:
        name (str): the name of the logger, typically the __name__ of the module
        level (Union[str,int]): the logging level to use
            (either a string like "INFO" or an int coming from the logging
            module, like logging.INFO)
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
