import asyncio
import json
import pickle
import os
import logging
import time

from adala.agents import Agent

from aiokafka import AIOKafkaConsumer
from celery import Celery, states
from celery.exceptions import Ignore
from server.utils import get_input_topic, get_output_topic, Settings
from server.handlers.result_handlers import ResultHandler


logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery(
    "worker", broker=REDIS_URL, backend=REDIS_URL, accept_content=["json", "pickle"]
)


@app.task(name="process_file", track_started=True, serializer="pickle")
def process_file(agent: Agent):
    # # Read data from a file and send it to the Kafka input topic
    asyncio.run(agent.environment.initialize())

    # run the agent
    asyncio.run(agent.arun())
    #
    # dump the output to a file
    asyncio.run(agent.environment.finalize())


@app.task(
    name="streaming_parent_task", track_started=True, bind=True, serializer="pickle"
)
def streaming_parent_task(
    self, agent: Agent, result_handler: ResultHandler, batch_size: int = 2
):
    """
    This task is used to launch the two tasks that are doing the real work, so that
    we store those two job IDs as metadata of this parent task, and be able to get
    the status of the entire job from one task ID
    """

    # Parent job ID is used for input/output topic names
    parent_job_id = self.request.id

    inference_task = process_file_streaming
    logger.info(f"Submitting task {inference_task.name} with agent {agent}")
    input_result = inference_task.delay(agent=agent, parent_job_id=parent_job_id)
    input_job_id = input_result.id
    logger.info(f"Task {inference_task.name} submitted with job_id {input_job_id}")

    result_handler_task = process_streaming_output
    logger.info(f"Submitting task {result_handler_task.name}")
    output_result = result_handler_task.delay(
        input_job_id=input_job_id,
        parent_job_id=parent_job_id,
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
    name="process_file_streaming", track_started=True, bind=True, serializer="pickle"
)
def process_file_streaming(self, agent: Agent, parent_job_id: str):
    # Set input and output topics using parent job ID
    agent.environment.kafka_input_topic = get_input_topic(parent_job_id)
    agent.environment.kafka_output_topic = get_output_topic(parent_job_id)

    # Run the agent
    asyncio.run(agent.arun())


async def async_process_streaming_output(
    input_job_id: str,
    parent_job_id: str,
    result_handler: ResultHandler,
    batch_size: int,
):
    logger.info(f"Polling for results {parent_job_id=}")

    topic = get_output_topic(parent_job_id)

    settings = Settings()

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    await consumer.start()
    logger.info(f"consumer started {parent_job_id=}")

    input_job_running = True

    while input_job_running:
        try:
            data = await consumer.getmany(timeout_ms=3000, max_records=batch_size)
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
        finally:
            job = process_file_streaming.AsyncResult(input_job_id)
            # TODO no way to recover here if connection to main app is lost, job will be stuck at "PENDING" so this will loop forever
            if job.status in ["SUCCESS", "FAILURE", "REVOKED"]:
                input_job_running = False
                logger.info(f"Input job done, stopping output job")
            else:
                logger.info(f"Input job still running, keeping output job running")

    await consumer.stop()


@app.task(
    name="process_streaming_output", track_started=True, bind=True, serializer="pickle"
)
def process_streaming_output(
    self,
    input_job_id: str,
    parent_job_id: str,
    result_handler: ResultHandler,
    batch_size: int,
):
    try:
        asyncio.run(
            async_process_streaming_output(
                input_job_id, parent_job_id, result_handler, batch_size
            )
        )
    except Exception as e:
        # Set own status to failure
        self.update_state(state=states.FAILURE)

        logger.error(msg=e)

        # Ignore the task so no other state is recorded
        raise Ignore()
