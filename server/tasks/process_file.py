import asyncio
import json
import pickle
import os
import logging

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
    name="process_file_streaming", track_started=True, bind=True, serializer="pickle"
)
def process_file_streaming(self, agent: Agent):
    # Get own job ID to set Consumer topic accordingly
    job_id = self.request.id
    agent.environment.kafka_input_topic = get_input_topic(job_id)
    agent.environment.kafka_output_topic = get_output_topic(job_id)

    # Run the agent
    asyncio.run(agent.arun())


async def async_process_streaming_output(
    input_job_id: str, result_handler: ResultHandler, batch_size: int
):
    logger.info(f"Polling for results {input_job_id=}")

    # FIXME switch to model_run_id or find a way to pass this in
    result_handler.set_job_id(input_job_id)

    topic = get_output_topic(input_job_id)
    settings = Settings()

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    await consumer.start()
    logger.info(f"consumer started {input_job_id=}")

    input_job_running = True

    while input_job_running:
        try:
            data = await consumer.getmany(timeout_ms=3000, max_records=batch_size)
            for tp, messages in data.items():
                if messages:
                    logger.info(f"Handling {messages=} in topic {tp.topic}")
                    data = [msg.value for msg in messages]
                    result_handler(data)
                    logger.info(f"Handled {len(messages)} messages in topic {tp.topic}")
                else:
                    logger.info(f"No messages in topic {tp.topic}")
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
    self, job_id: str, result_handler: ResultHandler, batch_size: int = 2
):
    try:
        asyncio.run(async_process_streaming_output(job_id, result_handler, batch_size))
    except Exception as e:
        # Set own status to failure
        self.update_state(state=states.FAILURE)

        logger.log(level=logging.ERROR, msg=e)

        # Ignore the task so no other state is recorded
        # TODO check if this updates state to Ignored, or keeps Failed
        raise Ignore()
