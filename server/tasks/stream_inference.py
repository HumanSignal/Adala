import asyncio
import json
import os
import psutil
import time
import traceback

from adala.agents import Agent

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import UnknownTopicOrPartitionError
from celery import Celery
from celery.signals import worker_process_shutdown, worker_process_init
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

settings = Settings()

app = Celery(
    "worker",
    broker=settings.redis.to_url(),
    backend=settings.redis.to_url(),
    accept_content=["json", "pickle"],
    broker_connection_retry_on_startup=True,
    worker_max_memory_per_child=settings.celery_worker_max_memory_per_child_kb,
    **{f"redis_{k}": v for k, v in settings.redis.to_kwargs().items()},
)


@worker_process_init.connect
def worker_process_init_handler(**kwargs):
    """Called when a worker process starts."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(
        f"Worker process starting. PID: {os.getpid()}, "
        f"Memory RSS: {mem_info.rss / 1024 / 1024:.2f}MB"
    )


@worker_process_shutdown.connect
def worker_process_shutdown_handler(**kwargs):
    """Called when a worker process shuts down."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(
        f"Worker process shutting down. PID: {os.getpid()}, "
        f"Memory RSS: {mem_info.rss / 1024 / 1024:.2f}MB"
    )


def parent_job_error_handler(self, exc, task_id, args, kwargs, einfo):
    """
    This function will be called if streaming_parent_task fails, to ensure that we cleanup any left over Kafka topics.
    """
    parent_job_id = task_id
    input_topic_name = get_input_topic_name(parent_job_id)
    output_topic_name = get_output_topic_name(parent_job_id)
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)


async def run_streaming(
    agent: Agent, result_handler: ResultHandler, batch_size: int, output_topic_name: str
):
    """
    This function is used to launch the two streaming tasks:
    - async_process_streaming_input: reads from the input topic and runs the agent
    - async_process_streaming_output: reads from the output topic and handles the results
    """
    input_task_done = asyncio.Event()
    async with asyncio.TaskGroup() as task_group:
        task_group.create_task(async_process_streaming_input(input_task_done, agent))
        task_group.create_task(
            async_process_streaming_output(
                input_task_done, output_topic_name, result_handler, batch_size
            )
        )


@app.task(
    name="streaming_parent_task",
    track_started=True,
    bind=True,
    serializer="pickle",
    on_failure=parent_job_error_handler,
    task_time_limit=settings.task_time_limit_sec,
    task_ignore_result=True,
    task_store_errors_even_if_ignored=True,
)
def streaming_parent_task(
    self, agent: Agent, result_handler: ResultHandler, batch_size: int = 1
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

    # Run the input and output streaming tasks
    asyncio.run(run_streaming(agent, result_handler, batch_size, output_topic_name))

    # Override default agent kafka settings
    agent.environment.kafka_input_topic = input_topic_name
    agent.environment.kafka_output_topic = output_topic_name
    agent.environment.timeout_ms = settings.kafka.input_consumer_timeout_ms

    # clean up kafka topics
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)

    logger.info("Both input and output jobs complete")


async def async_process_streaming_input(input_task_done: asyncio.Event, agent: Agent):
    try:

        # Override more kafka settings
        # these kwargs contain the SSL context, so must be done in the same context as initialize() to avoid de/serialization issues
        agent.environment.kafka_kwargs = settings.kafka.to_kafka_kwargs()

        # start up kaka producer and consumer
        await agent.environment.initialize()
        # Run the agent
        await agent.arun()
        input_task_done.set()
        # shut down kaka producer and consumer
        await agent.environment.finalize()
    except Exception as e:
        logger.error(
            f"Error in async_process_streaming_input: {e}. Traceback: {traceback.format_exc()}"
        )
        input_task_done.set()
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

    timeout_ms = settings.kafka.output_consumer_timeout_ms

    # Retry to workaround race condition of topic creation
    retries = 5
    while retries > 0:
        try:
            consumer = AIOKafkaConsumer(
                output_topic_name,
                bootstrap_servers=settings.kafka.bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                max_partition_fetch_bytes=3000000,
                # enable_auto_commit=False, # Turned off as its not supported without group ID
                # group_id=output_topic_name, # No longer using group ID as of DIA-1584 - unclear details but causes problems
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
                # messages is a list of ConsumerRecord
                if messages:
                    # batches is a list of lists
                    batches = [msg.value for msg in messages]
                    # records is a list of records to send to LSE
                    for records in batches:
                        logger.info(
                            f"Processing messages in output job {topic=} number of messages: {len(records)}"
                        )
                        result_handler(records)
                        logger.info(
                            f"Processed messages in output job {topic=} number of messages: {len(records)}"
                        )
                else:
                    logger.info(f"Consumer pulled data, but no messages in {topic=}")

            if not data:
                logger.info(f"Consumer pulled no data from {output_topic_name=}")

    # cleans up after any exceptions raised here as well as asyncio.CancelledError resulting from failure in async_process_streaming_input
    finally:
        logger.info(
            "No more data in output job and input job is done, stopping output job"
        )
        await consumer.stop()
