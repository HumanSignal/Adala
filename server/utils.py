from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union
from pathlib import Path
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


class Settings(BaseSettings):
    """
    Can hardcode settings here, read from env file, or pass as env vars
    https://docs.pydantic.dev/latest/concepts/pydantic_settings/#field-value-priority
    """

    kafka_bootstrap_servers: Union[str, List[str]]
    kafka_retention_ms: int

    model_config = SettingsConfigDict(
        # have to use an absolute path here so celery workers can find it
        env_file=(Path(__file__).parent / ".env"),
    )


def get_input_topic(job_id: str):
    topic_name = f"adala-input-{job_id}"

    # same logic as get_output_topic

    settings = Settings()
    bootstrap_servers = settings.kafka_bootstrap_servers
    retention_ms = settings.kafka_retention_ms

    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers, client_id="topic_creator"
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
        pass

    return topic_name


def get_output_topic(job_id: str):
    topic_name = f"adala-output-{job_id}"

    # same logic as get_input_topic

    settings = Settings()
    bootstrap_servers = settings.kafka_bootstrap_servers
    retention_ms = settings.kafka_retention_ms

    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers, client_id="topic_creator"
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
        pass

    return topic_name


def delete_topic(topic_name: str):
    # unused for now
    settings = Settings()
    bootstrap_servers = settings.kafka_bootstrap_servers

    admin_client = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers, client_id="topic_deleter"
    )

    admin_client.delete_topics(topics=[topic_name])
