import asyncio
import json
import pickle
import os
import logging

from aiokafka import AIOKafkaConsumer
from celery import Celery, states
from celery.exceptions import Ignore
from server.utils import get_input_topic, get_output_topic, ResultHandler, Settings


logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)


@app.task(name="process_file", track_started=True)
def process_file(serialized_agent: bytes):
    # Load the agent
    agent = pickle.loads(serialized_agent)
    # # Read data from a file and send it to the Kafka input topic
    asyncio.run(agent.environment.initialize())

    # run the agent
    asyncio.run(agent.arun())
    #
    # dump the output to a file
    asyncio.run(agent.environment.finalize())


@app.task(name="process_file_streaming", track_started=True, bind=True)
def process_file_streaming(self, serialized_agent: bytes):
    # Load the agent
    agent = pickle.loads(serialized_agent)

    # Get own job ID to set Consumer topic accordingly
    job_id = self.request.id
    agent.environment.kafka_input_topic = get_input_topic(job_id)
    agent.environment.kafka_output_topic = get_output_topic(job_id)

    # Run the agent
    asyncio.run(agent.arun())


async def async_process_streaming_output(input_job_id: str, result_handler: str, batch_size: int):
    logger.info(f"Polling for results {input_job_id=}")

    try:
        result_handler = ResultHandler.__dict__[result_handler]
    except KeyError as e:
        logger.error(f"{result_handler} is not a valid ResultHandler")
        raise e

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
                    result_handler(messages)
                else:
                    logger.info(f"No messages in topic {tp.topic}")
        finally:
            job = process_file_streaming.AsyncResult(input_job_id)
            if job.status in ['SUCCESS', 'FAILURE', 'REVOKED']:
                input_job_running = False
                logger.info(f"Input job done, stopping output job")
            else:
                logger.info(f"Input job still running, keeping output job running")

    await consumer.stop()


@app.task(name="process_streaming_output", track_started=True, bind=True)
def process_streaming_output(self, job_id: str, result_handler: str, batch_size: int = 2):
    try:
        asyncio.run(async_process_streaming_output(job_id, result_handler, batch_size))
    except KeyError:
        # Set own status to failure
        self.update_state(state=states.FAILURE)

        # Ignore the task so no other state is recorded
        # TODO check if this updates state to Ignored, or keeps Failed
        raise Ignore()
