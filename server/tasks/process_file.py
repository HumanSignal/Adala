import asyncio
import pickle
import os
import logging
from celery import Celery
from utils import get_input_topic, get_output_topic


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
