from urllib.parse import quote, urlparse, urlunparse, parse_qsl, urlencode
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union, Optional, Literal, Dict, Any
import logging
import os
from pathlib import Path
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError
from aiokafka.helpers import create_ssl_context
import asyncio

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(__name__)


class RedisSettings(BaseModel):
    """
    Redis settings including authentication and SSL options.
    """

    url: str = "redis://localhost:6379/0"
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    model_config = ConfigDict(
        extra="allow",
    )

    @property
    def ssl(self) -> bool:
        return self.ssl_ca_certs is not None

    def to_kwargs(self) -> Dict[str, Any]:
        """
        Kwargs that cannot be encoded in the url
        Right now none are set by default, but free to pass in at runtime
        """
        return self.model_dump(
            exclude_none=True,
            exclude=[
                "url",
                "username",
                "password",
                "ssl_cert_reqs",
                "ssl_ca_certs",
                "ssl_certfile",
                "ssl_keyfile",
            ],
        )

    def to_url(self) -> str:
        """
        Convert the RedisSettings object to a URL string.
        Params passed in separately take precedence over those in the URL.
        """
        parts = urlparse(self.url)
        if self.username or self.password:
            username = self.username or parts.username or ""
            password = self.password or parts.password or ""
            domain = parts.netloc.split("@")[-1]
            parts = parts._replace(
                netloc=f"{quote(username)}:{quote(password)}@{domain}"
            )

        # Convert query string to dict
        query_dict = dict(parse_qsl(parts.query))

        # Update with kwargs that can be encoded in the url
        if self.ssl:
            kwargs_to_update_query = self.model_dump(
                exclude_none=True,
                include=[
                    "ssl_cert_reqs",
                    "ssl_ca_certs",
                    "ssl_certfile",
                    "ssl_keyfile",
                ],
            )
            query_dict.update(kwargs_to_update_query)

        # Convert back to query string
        parts = parts._replace(query=urlencode(query_dict, doseq=False))

        return urlunparse(parts)


class KafkaSettings(BaseModel):
    """
    Kafka settings including authentication and SSL options.
    """

    # for topics
    retention_ms: int = 18000000  # 300 minutes

    # for consumers
    input_consumer_timeout_ms: int = 2500  # 2.5 seconds
    output_consumer_timeout_ms: int = 1500  # 1.5 seconds

    # for producers and consumers
    bootstrap_servers: Union[str, List[str]] = "localhost:9093"
    security_protocol: Literal["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"] = (
        "PLAINTEXT"
    )

    # SSL parameters
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_cert_password: Optional[str] = None

    # SASL parameters
    # NOTE: may want to add other SASL mechanisms SCRAM-SHA-256, SCRAM-SHA-512, OAUTHBEARER
    sasl_mechanism: Optional[Literal["PLAIN"]] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None

    model_config = ConfigDict(
        extra="allow",
    )

    def to_kafka_kwargs(self) -> Dict[str, Any]:
        """
        Convert the KafkaSettings object to kwargs for AIOKafkaProducer/Consumer/AdminClient.
        These are common kwargs for all Kafka objects; usage-specific kwargs are passed in separately.
        """
        kwargs = self.model_dump(include=["bootstrap_servers", "security_protocol"])

        # Add SSL parameters if using SSL
        if self.security_protocol in ["SSL", "SASL_SSL"]:
            ssl_context = create_ssl_context(
                cafile=self.ssl_cafile,
                certfile=self.ssl_certfile,
                keyfile=self.ssl_keyfile,
                password=self.ssl_cert_password,
            )
            kwargs["ssl_context"] = ssl_context

        # Add SASL parameters if using SASL
        if self.security_protocol in ["SASL_PLAINTEXT", "SASL_SSL"]:
            if self.sasl_mechanism == "PLAIN":
                kwargs.update(
                    {
                        "sasl_mechanism": "PLAIN",
                        "sasl_plain_username": self.sasl_plain_username,
                        "sasl_plain_password": self.sasl_plain_password,
                    }
                )

        return kwargs


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
        env_nested_delimiter="_",  # allows env vars like REDIS_URL -> redis.url
        env_nested_max_split=1,  # allows env vars like REDIS_SSL_CERT_REQS -> redis.ssl_cert_reqs
    )


def get_input_topic_name(job_id: str):
    topic_name = f"adala-input-{job_id}"

    return topic_name


def get_output_topic_name(job_id: str):
    topic_name = f"adala-output-{job_id}"

    return topic_name


def ensure_topic(topic_name: str, num_partitions: int = 1):
    """Ensure a Kafka topic exists (sync version)"""
    try:
        # Try to run in current event loop if one exists
        loop = asyncio.get_running_loop()
        # If we're in an event loop, create a task
        task = asyncio.create_task(_ensure_topic_async(topic_name, num_partitions))
        # Wait for it to complete
        while not task.done():
            pass
        if task.exception():
            raise task.exception()
    except RuntimeError:
        # No event loop running, use asyncio.run
        asyncio.run(_ensure_topic_async(topic_name, num_partitions))


async def ensure_topic_async(topic_name: str, num_partitions: int = 1):
    """Ensure a Kafka topic exists (async version)"""
    await _ensure_topic_async(topic_name, num_partitions)


async def _ensure_topic_async(topic_name: str, num_partitions: int = 1):
    """Internal async function to ensure topic exists with correct partition count"""
    settings = Settings()
    kafka_kwargs = settings.kafka.to_kafka_kwargs()
    retention_ms = settings.kafka.retention_ms

    admin_client = AIOKafkaAdminClient(
        **kafka_kwargs,
        client_id="topic_creator",
    )

    try:
        await admin_client.start()

        # Check if topic exists and get its metadata
        topic_exists = False
        current_partitions = 0

        try:
            topic_metadata = await admin_client.describe_topics([topic_name])
            if topic_name in topic_metadata:
                topic_exists = True
                current_partitions = len(topic_metadata[topic_name].partitions)
                logger.info(
                    f"Topic {topic_name} exists with {current_partitions} partitions"
                )
        except Exception as e:
            logger.debug(f"Error checking topic {topic_name}: {e}")
            # Topic doesn't exist or other error, proceed with creation
            pass

        # If topic exists but has fewer partitions than required, delete and recreate
        if topic_exists and current_partitions < num_partitions:
            logger.warning(
                f"Topic {topic_name} has {current_partitions} partitions, but {num_partitions} required. Recreating..."
            )

            try:
                # Delete the existing topic
                await admin_client.delete_topics([topic_name])
                logger.info(f"Deleted topic {topic_name} with insufficient partitions")

                # Wait a bit for deletion to complete
                await asyncio.sleep(2)

                # Reset flag to create new topic
                topic_exists = False

            except UnknownTopicOrPartitionError:
                logger.warning(
                    f"Topic {topic_name} does not exist during deletion attempt"
                )
                topic_exists = False
            except Exception as e:
                logger.error(f"Error deleting topic {topic_name}: {e}")
                # Continue anyway, maybe we can still create it
                topic_exists = False

        # Create topic if it doesn't exist or was just deleted
        if not topic_exists or current_partitions < num_partitions:
            topic = NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=1,
                topic_configs={"retention.ms": str(retention_ms)},
            )

            try:
                await admin_client.create_topics([topic])
                logger.info(
                    f"Created topic {topic_name} with {num_partitions} partitions"
                )
            except TopicAlreadyExistsError:
                # Topic was created between our check and creation attempt
                logger.info(f"Topic {topic_name} already exists (created concurrently)")
                # Verify it has the right number of partitions
                try:
                    topic_metadata = await admin_client.describe_topics([topic_name])
                    if topic_name in topic_metadata:
                        final_partitions = len(topic_metadata[topic_name].partitions)
                        if final_partitions < num_partitions:
                            logger.error(
                                f"Topic {topic_name} was created with {final_partitions} partitions, but {num_partitions} required!"
                            )
                        else:
                            logger.info(
                                f"Topic {topic_name} has correct partition count: {final_partitions}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Error verifying final partition count for {topic_name}: {e}"
                    )
        else:
            logger.info(
                f"Topic {topic_name} already has sufficient partitions ({current_partitions} >= {num_partitions})"
            )

    finally:
        await admin_client.close()


def delete_topic(topic_name: str):
    """Delete a Kafka topic (sync version)"""
    try:
        # Try to run in current event loop if one exists
        loop = asyncio.get_running_loop()
        # If we're in an event loop, create a task
        task = asyncio.create_task(_delete_topic_async(topic_name))
        # Wait for it to complete
        while not task.done():
            pass
        if task.exception():
            raise task.exception()
    except RuntimeError:
        # No event loop running, use asyncio.run
        asyncio.run(_delete_topic_async(topic_name))


async def delete_topic_async(topic_name: str):
    """Delete a Kafka topic (async version)"""
    await _delete_topic_async(topic_name)


async def _delete_topic_async(topic_name: str):
    """Internal async function to delete topic"""
    settings = Settings()
    kafka_kwargs = settings.kafka.to_kafka_kwargs()

    admin_client = AIOKafkaAdminClient(
        **kafka_kwargs,
        client_id="topic_deleter",
    )

    try:
        await admin_client.start()
        try:
            await admin_client.delete_topics([topic_name])
            logger.info(f"Successfully deleted topic: {topic_name}")
        except UnknownTopicOrPartitionError:
            logger.warning(f"Topic {topic_name} does not exist and cannot be deleted.")
    finally:
        await admin_client.close()


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


async def ensure_worker_pool_topics():
    """Ensure all worker pool topics exist with appropriate partition counts"""
    topics_config = [
        (
            "worker_pool_input",
            50,
        ),  # Multiple partitions for load balancing across workers
        ("worker_pool_output", 1),  # Single partition for output
    ]

    for topic_name, num_partitions in topics_config:
        await ensure_topic_async(topic_name, num_partitions)
        logger.info(f"Ensured topic {topic_name} with {num_partitions} partitions")


async def ensure_worker_pool_input_topic():
    """Ensure worker pool input topic exists with exactly 50 partitions for proper load balancing"""
    topic_name = "worker_pool_input"
    required_partitions = 50

    await ensure_topic_async(topic_name, required_partitions)


def ensure_worker_pool_topics_sync():
    """Synchronous version of ensure_worker_pool_topics"""
    asyncio.run(ensure_worker_pool_topics())
