import asyncio
import pickle
import os
import logging
from celery import Celery

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app = Celery('worker', broker=REDIS_URL, backend=REDIS_URL)


@app.task(name='process_file')
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

