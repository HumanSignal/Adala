import asyncio
import json
import os
import logging
import time

from adala.agents import Agent

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import UnknownTopicOrPartitionError
from celery import Celery
from server.utils import (
    get_input_topic_name,
    get_output_topic_name,
    ensure_topic,
    delete_topic,
    Settings,
)
from server.handlers.result_handlers import ResultHandler


logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery(
    "worker", broker=REDIS_URL, backend=REDIS_URL, accept_content=["json", "pickle"]
)

settings = Settings()


def parent_job_error_handler(self, exc, task_id, args, kwargs, einfo):
    """
    This function will be called if streaming_parent_task fails, to ensure that we cleanup any left over
    Kafka topics.
    """
    parent_job_id = task_id
    input_topic_name = get_input_topic_name(parent_job_id)
    output_topic_name = get_output_topic_name(parent_job_id)
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)


@app.task(
    name="process_file",
    track_started=True,
    serializer="pickle",
    task_time_limit=settings.task_time_limit_sec,
)
def process_file(agent: Agent):
    # Override kafka_bootstrap_servers with value from settings
    agent.environment.kafka_bootstrap_servers = settings.kafka_bootstrap_servers

    # # Read data from a file and send it to the Kafka input topic
    asyncio.run(agent.environment.initialize())

    # run the agent
    asyncio.run(agent.arun())
    #
    # dump the output to a file
    asyncio.run(agent.environment.finalize())


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

    print(f"\n\n\nDOING SOMETHING WACK {parent_job_id}\n\n\n", flush=True)

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
    except asyncio.CancelledError as e:
        await agent.environment.finalize()
        raise e


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

    data = await consumer.getmany(timeout_ms=timeout_ms, max_records=batch_size)

    try:
        while not input_done.is_set():
            for tp, messages in data.items():
                if messages:
                    logger.debug(f"Handling {messages=} in topic {tp.topic}")
                    data = [msg.value for msg in messages]
                    result_handler(data)
                    logger.debug(
                        f"Handled {len(messages)} messages in topic {tp.topic}"
                    )
                else:
                    logger.debug(f"No messages in topic {tp.topic}")

            if not data:
                logger.info(f"No messages in any topic")

            # we are getting packets from the output topic here to check if its empty and continue processing if its not
            data = await consumer.getmany(timeout_ms=timeout_ms, max_records=batch_size)

    except asyncio.CancelledError as e:
        await consumer.stop()
        raise e

    finally:
        await consumer.stop()
