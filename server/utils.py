import sys

# fix for https://github.com/dpkp/kafka-python/issues/2412
if sys.version_info >= (3, 12, 0):
    import six

    sys.modules["kafka.vendor.six.moves"] = six.moves
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union
import logging
import os
from pathlib import Path
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError
import asyncio

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    """

    kafka_bootstrap_servers: Union[str, List[str]] = "localhost:9093"
    kafka_retention_ms: int = 18000000  # 300 minutes
    kafka_input_consumer_timeout_ms: int = 2500  # 2.5 seconds
    kafka_output_consumer_timeout_ms: int = 1500  # 1.5 seconds
    task_time_limit_sec: int = 60 * 60 * 6  # 6 hours
    # https://docs.celeryq.dev/en/v5.4.0/userguide/configuration.html#worker-max-memory-per-child
    celery_worker_max_memory_per_child_kb: int = 1024000  # 1GB

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

    async def _ensure_topic():
        admin_client = AIOKafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="topic_creator",
        )

        try:
            await admin_client.start()
            topic = NewTopic(
                name=topic_name,
                num_partitions=1,
                replication_factor=1,
                topic_configs={"retention.ms": str(retention_ms)},
            )

            try:
                await admin_client.create_topics([topic])
            except TopicAlreadyExistsError:
                # we shouldn't hit this case when KAFKA_CFG_AUTO_CREATE_TOPICS=false unless there is a legitimate name collision, so should raise here after testing
                pass
        finally:
            await admin_client.close()

    asyncio.run(_ensure_topic())


def delete_topic(topic_name: str):
    settings = Settings()
    bootstrap_servers = settings.kafka_bootstrap_servers

    async def _delete_topic():
        admin_client = AIOKafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="topic_deleter",
        )

        try:
            await admin_client.start()
            try:
                await admin_client.delete_topics([topic_name])
            except UnknownTopicOrPartitionError:
                logger.error(
                    f"Topic {topic_name} does not exist and cannot be deleted."
                )
        finally:
            await admin_client.close()

    asyncio.run(_delete_topic())


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
