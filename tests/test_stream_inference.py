import pytest
import os
import asyncio
from unittest.mock import AsyncMock, patch
from adala.environments.kafka import AsyncKafkaEnvironment
from server.tasks.stream_inference import run_streaming
from server.handlers.result_handlers import ResultHandler, LSEHandler
from adala.agents import Agent


TEST_AGENT = {
    "runtimes": {
        "default": {
            "type": "AsyncLiteLLMChatRuntime",
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 200,
            "temperature": 0,
            "batch_size": 100,
            "timeout": 10,
            "verbose": False,
        }
    },
    "environment": {
        "type": "AsyncKafkaEnvironment",
        "kafka_bootstrap_servers": "localhost:9092",
        "kafka_input_topic": "input_topic",
        "kafka_output_topic": "output_topic",
        "timeout_ms": 1000,
    },
    "skills": [
        {
            "type": "ClassificationSkill",
            "name": "ClassificationResult",
            "instructions": "",
            "input_template": "Classify sentiment of the input text: {input}",
            "field_schema": {
                "output": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                }
            },
        }
    ],
}

TEST_INPUT_DATA = [
    {"task_id": 100, "input": "I am happy"},
]
TEST_OUTPUT_DATA = [
    {
        "task_id": 100,
        "input": "I am happy",
        "output": "positive",
        "_completion_cost_usd": 3e-06,
        "_completion_tokens": 5,
        "_prompt_cost_usd": 1.365e-05,
        "_prompt_tokens": 91,
        "_total_cost_usd": 1.6649999999999998e-05,
    }
]


@pytest.fixture
def mock_kafka_consumer_input():
    with patch("adala.environments.kafka.AIOKafkaConsumer") as MockConsumer:
        mock_consumer = MockConsumer.return_value
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()
        mock_consumer.commit = AsyncMock()
        mock_consumer.getmany = AsyncMock(
            side_effect=[
                # first call return batch
                {"topic_partition": [AsyncMock(value=row) for row in TEST_INPUT_DATA]},
                # second call - end of the stream
                {},
            ]
        )
        yield mock_consumer


PRODUCER_SENT_DATA = asyncio.Event()


@pytest.fixture
def mock_kafka_consumer_output():

    async def getmany_side_effect(*args, **kwargs):
        await PRODUCER_SENT_DATA.wait()
        return {
            AsyncMock(topic="output_topic_partition"): [
                AsyncMock(value=TEST_OUTPUT_DATA)
            ]
        }

    with patch("server.tasks.stream_inference.AIOKafkaConsumer") as MockConsumer:
        mock_consumer = MockConsumer.return_value
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()
        mock_consumer.commit = AsyncMock()
        mock_consumer.getmany = AsyncMock(side_effect=getmany_side_effect)
        yield mock_consumer


@pytest.fixture
def mock_kafka_producer():
    with patch("adala.environments.kafka.AIOKafkaProducer") as MockProducer:
        mock_producer = MockProducer.return_value
        mock_producer.start = AsyncMock()
        mock_producer.stop = AsyncMock()

        async def send_and_wait_side_effect(*args, **kwargs):
            PRODUCER_SENT_DATA.set()
            return AsyncMock()

        async def send_side_effect(*args, **kwargs):
            PRODUCER_SENT_DATA.set()
            return AsyncMock()

        mock_producer.send_and_wait = AsyncMock(side_effect=send_and_wait_side_effect)
        mock_producer.send = AsyncMock(side_effect=send_side_effect)

        yield mock_producer


@pytest.fixture
def agent(mock_kafka_consumer_input, mock_kafka_producer):

    with patch.object(AsyncKafkaEnvironment, "initialize", AsyncMock()) as mock_init:
        agent = Agent(**TEST_AGENT)
        agent.environment.consumer = mock_kafka_consumer_input
        agent.environment.producer = mock_kafka_producer
        yield agent


@pytest.fixture
def mock_lse_client():
    with patch("server.handlers.result_handlers.LSEClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.check_connection.return_value = {"status": "UP"}
        yield mock_client


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_run_streaming(
    agent,
    mock_kafka_consumer_input,
    mock_kafka_producer,
    mock_kafka_consumer_output,
    mock_lse_client,
):

    result_handler = LSEHandler(
        api_key="api_key", url="http://fakeapp.humansignal.com", modelrun_id=123
    )

    # Call the run_streaming function
    await run_streaming(
        agent=agent,
        result_handler=result_handler,
        batch_size=1,
        output_topic_name="output_topic",
    )

    # Verify that producer is called with the correct amount of send_and_wait calls and data
    assert mock_kafka_producer.send_and_wait.call_count == 1
    mock_kafka_producer.send_and_wait.assert_any_call(
        "output_topic", value=TEST_OUTPUT_DATA
    )
