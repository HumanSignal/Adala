import asyncio
import json
import pickle
import os
import logging

from aiokafka import AIOKafkaConsumer
from celery import Celery
from server.utils import dummy_handler, get_input_topic, get_output_topic, Settings
from typing import Callable


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


async def async_process_streaming_output(job_id: str, batch_size: int):
    logger.info(f"Polling for results {job_id=}")

    topic = get_output_topic(job_id)
    settings = Settings()

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest"
    )
    await consumer.start()
    logger.info(f"consumer started {job_id=}")

    try:
        data = await consumer.getmany(timeout_ms=30_000, max_records=batch_size)
        logger.info(f"got batch {data=}")
        for tp, messages in data.items():
            if messages:
                dummy_handler(messages)
            else:
                logger.info(f"No messages in topic {tp.topic}")
    finally:
        await consumer.stop()


@app.task(name="process_streaming_output", track_started=True)
def process_streaming_output(job_id: str, batch_size: int = 2):
    asyncio.run(async_process_streaming_output(job_id, batch_size))