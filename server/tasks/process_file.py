import asyncio
import json
import os
import time

from adala.agents import Agent

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import UnknownTopicOrPartitionError
from celery import Celery
from server.handlers.result_handlers import ResultHandler
from server.utils import (
    Settings,
    delete_topic,
    ensure_topic,
    get_input_topic_name,
    get_output_topic_name,
    init_logger,
)

logger = init_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    accept_content=["json", "pickle"],
    broker_connection_retry_on_startup=True,
)

settings = Settings()


def parent_job_error_handler(self, exc, task_id, args, kwargs, einfo):
    """
    This function will be called if streaming_parent_task fails, to ensure that we cleanup any left over Kafka topics.
    """
    parent_job_id = task_id
    input_topic_name = get_input_topic_name(parent_job_id)
    output_topic_name = get_output_topic_name(parent_job_id)
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)


@app.task(
    name="streaming_parent_task",
    track_started=True,
    bind=True,
    serializer="pickle",
    on_failure=parent_job_error_handler,
    task_time_limit=settings.task_time_limit_sec,
)
def streaming_parent_task(
    self, agent: Agent, result_handler: ResultHandler, batch_size: int = 10
):
    """
    This task is used to launch the two tasks that are doing the real work, so that
    we store those two job IDs as metadata of this parent task, and be able to get
    the status of the entire job from one task ID
    """

    # Parent job ID is used for input/output topic names
    parent_job_id = self.request.id

    # create kafka topics
    input_topic_name = get_input_topic_name(parent_job_id)
    ensure_topic(input_topic_name)
    output_topic_name = get_output_topic_name(parent_job_id)
    ensure_topic(output_topic_name)

    # Override default agent kafka settings
    agent.environment.kafka_bootstrap_servers = settings.kafka_bootstrap_servers
    agent.environment.kafka_input_topic = input_topic_name
    agent.environment.kafka_output_topic = output_topic_name
    agent.environment.timeout_ms = settings.kafka_input_consumer_timeout_ms

    async def run_streaming():
        input_task_done = asyncio.Event()
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(
                async_process_streaming_input(input_task_done, agent)
            )
            task_group.create_task(
                async_process_streaming_output(
                    input_task_done, output_topic_name, result_handler, batch_size
                )
            )

    asyncio.run(run_streaming())

    # clean up kafka topics
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)

    logger.info("Both input and output jobs complete")


async def async_process_streaming_input(input_task_done: asyncio.Event, agent: Agent):
    try:
        # start up kaka producer and consumer
        await agent.environment.initialize()
        # Run the agent
        await agent.arun()
        input_task_done.set()
        # shut down kaka producer and consumer
        await agent.environment.finalize()
    # cleans up after any exceptions raised here as well as asyncio.CancelledError resulting from failure in async_process_streaming_output
    finally:
        await agent.environment.finalize()


async def async_process_streaming_output(
    input_done: asyncio.Event,
    output_topic_name,
    result_handler: ResultHandler,
    batch_size: int,
):
    logger.info(f"Polling for results {output_topic_name=}")

    timeout_ms = settings.kafka_output_consumer_timeout_ms

    # Retry to workaround race condition of topic creation
    retries = 5
    while retries > 0:
        try:
            consumer = AIOKafkaConsumer(
                output_topic_name,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
            )
            await consumer.start()
            logger.info(f"consumer started {output_topic_name=}")
            break
        except UnknownTopicOrPartitionError as e:
            logger.error(msg=e)
            logger.info(f"Retrying to create consumer with topic {output_topic_name}")

            await consumer.stop()
            retries -= 1
            time.sleep(1)

    try:
        while not input_done.is_set():
            data = await consumer.getmany(timeout_ms=timeout_ms, max_records=batch_size)
            for topic_partition, messages in data.items():
                topic = topic_partition.topic
                if messages:
                    logger.debug(f"Handling {messages=} in {topic=}")
                    data = [msg.value for msg in messages]
                    result_handler(data)
                    logger.debug(f"Handled {len(messages)} messages in {topic=}")
                else:
                    logger.debug(f"No messages in topic {topic=}")

            if not data:
                logger.info("No messages in any topic")

    # cleans up after any exceptions raised here as well as asyncio.CancelledError resulting from failure in async_process_streaming_input
    finally:
        await consumer.stop()
