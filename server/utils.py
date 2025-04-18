from urllib.parse import quote, urlparse, urlunparse, parse_qsl, urlencode
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union, Optional, Literal, Dict, Any
import logging
import os
from pathlib import Path
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError, UnknownTopicOrPartitionError
from aiokafka.helpers import create_ssl_context
import asyncio
import ssl

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
        kwargs_to_update_query = self.model_dump(
            include=[
                "ssl",
                "ssl_cert_reqs",
                "ssl_ca_certs",
                "ssl_certfile",
                "ssl_keyfile",
            ]
        )
        query_dict.update(kwargs_to_update_query)

        # Convert back to query string
        parts._replace(query=urlencode(query_dict, doseq=False))

        return urlunparse(parts)


class KafkaSettings(BaseSettings):
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
    security_protocol: Literal["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"] = "PLAINTEXT"
    
    # SSL parameters
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # SASL parameters
    # NOTE: may want to add other SASL mechanisms SCRAM-SHA-256, SCRAM-SHA-512, OAUTHBEARER
    sasl_mechanism: Optional[Literal["PLAIN", "GSSAPI"]] = None
    sasl_plain_username: Optional[str] = None
    sasl_plain_password: Optional[str] = None
    sasl_kerberos_service_name: Optional[str] = "kafka"
    sasl_kerberos_domain_name: Optional[str] = None

    def to_kafka_kwargs(self) -> Dict[str, Any]:
        """
        Convert the KafkaSettings object to kwargs for AIOKafkaProducer/Consumer/AdminClient.
        These are common kwargs for all Kafka objects; usage-specific kwargs are passed in separately.
        """
        kwargs = self.model_dump(include=["bootstrap_servers", "security_protocol"])

        # Add SSL parameters if using SSL
        if self.security_protocol in ["SSL", "SASL_SSL"]:
            ssl_context = create_ssl_context(
                cafile=self.ssl_ca_certs,
                certfile=self.ssl_certfile,
                keyfile=self.ssl_keyfile,
            )
            if self.ssl_cert_reqs:
                ssl_context.verify_mode = getattr(ssl, f"CERT_{self.ssl_cert_reqs.upper()}")
            kwargs["ssl_context"] = ssl_context

        # Add SASL parameters if using SASL
        if self.security_protocol in ["SASL_PLAINTEXT", "SASL_SSL"]:
            if self.sasl_mechanism == "PLAIN":
                kwargs.update({
                    "sasl_mechanism": "PLAIN",
                    "sasl_plain_username": self.sasl_plain_username,
                    "sasl_plain_password": self.sasl_plain_password,
                })
            elif self.sasl_mechanism == "GSSAPI":
                kwargs.update({
                    "sasl_mechanism": "GSSAPI",
                    "sasl_kerberos_service_name": self.sasl_kerberos_service_name,
                    "sasl_kerberos_domain_name": self.sasl_kerberos_domain_name,
                })

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


def ensure_topic(topic_name: str):
    settings = Settings()
    kafka_kwargs = settings.kafka.to_kafka_kwargs()
    retention_ms = settings.kafka.retention_ms

    async def _ensure_topic():
        admin_client = AIOKafkaAdminClient(
            **kafka_kwargs,
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
    kafka_kwargs = settings.kafka.to_kafka_kwargs()

    async def _delete_topic():
        admin_client = AIOKafkaAdminClient(
            **kafka_kwargs,
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
