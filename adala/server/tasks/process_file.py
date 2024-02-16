import asyncio
import json
import os
from celery import Celery
from typing import List
from adala.agents import Agent
from adala.environments.kafka import FileStreamAsyncKafkaEnvironment

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_INPUT_TOPIC = os.getenv('KAFKA_INPUT_TOPIC', 'input')
KAFKA_OUTPUT_TOPIC = os.getenv('KAFKA_OUTPUT_TOPIC', 'output')
app = Celery('worker', broker=REDIS_URL, backend=REDIS_URL)


@app.task(name='process_file')
def process_file(
    input_file: str,
    serialized_agent: str,
    output_file: str,
    error_file: str,
    output_columns: List[str]
):
    agent = json.loads(serialized_agent)
    env = FileStreamAsyncKafkaEnvironment(
        kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        kafka_input_topic=KAFKA_INPUT_TOPIC,
        kafka_output_topic=KAFKA_OUTPUT_TOPIC
    )

    # Define an agent
    agent = Agent(**json.loads(serialized_agent))
    agent.environment = env

    # Read data from a file and send it to the Kafka input topic
    asyncio.run(env.read_from_file(input_file))

    # run the agent
    asyncio.run(agent.arun())

    # dump the output to a file
    asyncio.run(env.write_to_file(output_file, output_columns))
