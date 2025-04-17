import sys

# fix for https://github.com/dpkp/kafka-python/issues/2412
if sys.version_info >= (3, 12, 0):
    import six

    sys.modules["kafka.vendor.six.moves"] = six.moves
from urllib.parse import quote, urlparse, urlunparse, parse_qsl, urlencode
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union, Optional
import logging
import os
from pathlib import Path
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError
import asyncio

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(__name__)


class RedisSettings(BaseSettings):
    """
    Redis settings including authentication and SSL options.
    """
    url: str = "redis://localhost:6379/0"
    socket_connect_timeout: int = 1
    username: Optional[str] = None
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    def to_url(self) -> str:
        """
        Convert the RedisSettings object to a URL string.
        Params passed in separately take precedence over those in the URL.
        """
        parts = urlparse(self.url)
        if self.username:
            parts.username = quote(self.username)
        if self.password:
            parts.password = quote(self.password)

        # Convert query string to dict
        query_dict = dict(parse_qsl(parts.query))
        
        # Update with new kwargs
        kwargs_to_update_query = self.model_dump(include=['ssl', 'ssl_cert_reqs', 'ssl_ca_certs', 'ssl_certfile', 'ssl_keyfile'])
        query_dict.update(kwargs_to_update_query)
        
        # Convert back to query string
        parts._replace(query=urlencode(query_dict, doseq=False))

        return urlunparse(parts)
        

class KafkaSettings(BaseSettings):
    """
    Kafka settings including authentication and SSL options.
    """
    bootstrap_servers: Union[str, List[str]] = "localhost:9093"
    retention_ms: int = 18000000  # 300 minutes
    input_consumer_timeout_ms: int = 2500  # 2.5 seconds
    output_consumer_timeout_ms: int = 1500  # 1.5 seconds


class Settings(BaseSettings):
    """
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    """

    kafka: KafkaSettings = KafkaSettings()
    task_time_limit_sec: int = 60 * 60 * 6  # 6 hours
    # https://docs.celeryq.dev/en/v5.4.0/userguide/configuration.html#worker-max-memory-per-child
    celery_worker_max_memory_per_child_kb: int = 1024000  # 1GB
    redis: RedisSettings = RedisSettings()

    model_config = SettingsConfigDict(
        # have to use an absolute path here so celery workers can find it
        env_file=(Path(__file__).parent / ".env"),
        env_nested_delimiter='_'  # allows REDIS_SSL_ENABLED=true in env
    )


def get_input_topic_name(job_id: str):
    topic_name = f"adala-input-{job_id}"

    return topic_name


def get_output_topic_name(job_id: str):
    topic_name = f"adala-output-{job_id}"

    return topic_name


def ensure_topic(topic_name: str):
    settings = Settings()
    bootstrap_servers = settings.kafka.bootstrap_servers
    retention_ms = settings.kafka.retention_ms

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
    bootstrap_servers = settings.kafka.bootstrap_servers

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
