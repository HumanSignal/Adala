import asyncio
import json
import pickle
import os
import logging
import time

from adala.agents import Agent

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import UnknownTopicOrPartitionError
from celery import Celery, states
from celery.exceptions import Ignore
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

    Also attempts to stop input/output jobs if their IDs are available
    """
    parent_job_id = task_id
    input_topic_name = get_input_topic_name(parent_job_id)
    output_topic_name = get_output_topic_name(parent_job_id)
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)

    parent_job = streaming_parent_task.AsyncResult(parent_job_id)
    if (
        parent_job.info is None
        or type(parent_job.info)
        != dict  # In some failure cases e.g. an exception is thrown, job.info will be a string, causing the next line to crash
        or "input_job_id" not in parent_job.info
        or "output_job_id" not in parent_job.info
    ):
        logger.warning(
            "Parent task does not contain input job ID and/or output_job_id - unable to stop input/output jobs"
        )
    else:
        input_job_id = parent_job.info["input_job_id"]
        output_job_id = parent_job.info["output_job_id"]
        input_job = process_file_streaming.AsyncResult(input_job_id)
        output_job = process_streaming_output.AsyncResult(output_job_id)
        input_job.revoke()
        output_job.revoke()


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

    inference_task = process_file_streaming
    logger.info(f"Submitting task {inference_task.name} with agent {agent}")
    input_result = inference_task.delay(agent=agent)
    input_job_id = input_result.id
    logger.info(f"Task {inference_task.name} submitted with job_id {input_job_id}")

    result_handler_task = process_streaming_output
    logger.info(f"Submitting task {result_handler_task.name}")
    output_result = result_handler_task.delay(
        input_job_id=input_job_id,
        output_topic_name=output_topic_name,
        result_handler=result_handler,
        batch_size=batch_size,
    )
    output_job_id = output_result.id
    logger.info(
        f"Task {result_handler_task.name} submitted with job_id {output_job_id}"
    )

    # Store input and output job IDs in parent task metadata
    # Need to pass state as well otherwise its overwritten to None
    self.update_state(
        state=states.STARTED,
        meta={"input_job_id": input_job_id, "output_job_id": output_job_id},
    )

    input_job = process_file_streaming.AsyncResult(input_job_id)
    output_job = process_streaming_output.AsyncResult(output_job_id)

    terminal_statuses = ["SUCCESS", "FAILURE", "REVOKED"]

    while (
        input_job.status not in terminal_statuses
        or output_job.status not in terminal_statuses
    ):
        time.sleep(1)

    # clean up kafka topics
    delete_topic(input_topic_name)
    delete_topic(output_topic_name)

    logger.info("Both input and output jobs complete")

    # Update parent task status to SUCCESS and pass metadata again
    # otherwise its overwritten to None
    self.update_state(
        state=states.SUCCESS,
        meta={"input_job_id": input_job_id, "output_job_id": output_job_id},
    )

    # This makes it so Celery doesnt update the tasks state again, which would wipe out the custom metadata we added
    # It will retain that state we set above
    raise Ignore()


@app.task(
    name="process_file_streaming",
    track_started=True,
    serializer="pickle",
    task_time_limit=settings.task_time_limit_sec,
)
def process_file_streaming(agent: Agent):
    # agent's kafka_bootstrap servers and kafka topics should be set in parent task

    # need to keep these in the same event loop
    async def run_fn():
        # start up kaka producer and consumer
        await agent.environment.initialize()
        # Run the agent
        await agent.arun()
        # shut down kaka producer and consumer
        await agent.environment.finalize()

    asyncio.run(run_fn())


async def async_process_streaming_output(
    input_job_id: str,
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

    input_job_running = True

    data = await consumer.getmany(timeout_ms=timeout_ms, max_records=batch_size)

    while input_job_running:
        for tp, messages in data.items():
            if messages:
                logger.debug(f"Handling {messages=} in topic {tp.topic}")
                data = [msg.value for msg in messages]
                result_handler(data)
                logger.debug(f"Handled {len(messages)} messages in topic {tp.topic}")
            else:
                logger.debug(f"No messages in topic {tp.topic}")

        if not data:
            logger.info(f"No messages in any topic")

        job = process_file_streaming.AsyncResult(input_job_id)
        # we are getting packets from the output topic here to check if its empty and continue processing if its not
        data = await consumer.getmany(timeout_ms=timeout_ms, max_records=batch_size)
        # TODO no way to recover here if connection to main app is lost, job will be stuck at "PENDING" so this will loop forever
        if job.status in ["SUCCESS", "FAILURE", "REVOKED"] and len(data.items()) == 0:
            input_job_running = False
            logger.info(f"Input job done, stopping output job")
        else:
            logger.info(f"Input job still running, keeping output job running")

    await consumer.stop()


@app.task(
    name="process_streaming_output",
    track_started=True,
    bind=True,
    serializer="pickle",
    task_time_limit=settings.task_time_limit_sec,
)
def process_streaming_output(
    self,
    input_job_id: str,
    output_topic_name: str,
    result_handler: ResultHandler,
    batch_size: int,
):
    try:
        asyncio.run(
            async_process_streaming_output(
                input_job_id, output_topic_name, result_handler, batch_size
            )
        )
    except Exception as e:
        # Set own status to failure
        self.update_state(state=states.FAILURE)

        logger.error(msg=e)

        # Ignore the task so no other state is recorded
        raise Ignore()
