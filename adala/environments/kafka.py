import logging
import abc
import boto3
import json
import asyncio
import aiohttp
import math
from csv import DictReader, DictWriter
from typing import Dict, Union, List, Optional, Iterable, Any
from io import StringIO
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from adala.utils.internal_data import InternalDataFrame
from adala.environments import Environment, AsyncEnvironment, EnvironmentFeedback
from adala.skills import SkillSet
from adala.utils.logs import print_text

logger = logging.getLogger(__name__)


class AsyncKafkaEnvironment(AsyncEnvironment):
    """
    Represents an asynchronous Kafka environment:
    - agent can retrieve data batch by batch from the input topic
    - agent can return its predictions to the output topic

    Attributes:
        kafka_kwargs (Dict[str, Any]): The Kafka kwargs, including at least bootstrap_servers.
        kafka_input_topic (str): The Kafka input topic.
        kafka_output_topic (str): The Kafka output topic.
        timeout_ms (int): The timeout for the Kafka consumer.
    """

    # these are mandatory, but should be set by server
    kafka_kwargs: Optional[Dict[str, Any]] = None
    kafka_input_topic: Optional[str] = None
    kafka_output_topic: Optional[str] = None
    timeout_ms: Optional[int] = None

    # these are set in initialize()
    consumer: Optional[AIOKafkaConsumer] = None
    producer: Optional[AIOKafkaProducer] = None

    async def initialize(self):
        assert (
            self.kafka_kwargs is not None and 'bootstrap_servers' in self.kafka_kwargs
        ), "missing initialization for kafka_kwargs"
        assert (
            self.kafka_input_topic is not None
        ), "missing initialization for kafka_input_topic"
        assert (
            self.kafka_output_topic is not None
        ), "missing initialization for kafka_output_topic"
        assert self.timeout_ms is not None, "missing initialization for timeout_ms"

        self.consumer = AIOKafkaConsumer(
            self.kafka_input_topic,
            **self.kafka_kwargs,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            max_partition_fetch_bytes=3000000,
            # enable_auto_commit=False, # Turned off as its not supported without group ID
            # group_id=output_topic_name, # No longer using group ID as of DIA-1584 - unclear details but causes problems
        )
        await self.consumer.start()

        self.producer = AIOKafkaProducer(
            **self.kafka_kwargs,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            max_request_size=3000000,
            acks="all",  # waits for all replicas to respond that they have written the message
        )
        await self.producer.start()

    async def finalize(self):
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()

    async def get_feedback(
        self,
        skills: SkillSet,
        predictions: InternalDataFrame,
        num_feedbacks: Optional[int] = None,
    ) -> EnvironmentFeedback:
        raise NotImplementedError("Feedback is not supported in Kafka environment")

    async def restore(self):
        raise NotImplementedError("Restore is not supported in Kafka environment")

    async def save(self):
        raise NotImplementedError("Save is not supported in Kafka environment")

    # TODO replace this with
    # https://aiokafka.readthedocs.io/en/stable/api.html#aiokafka.AIOKafkaProducer.send_batch
    async def message_sender(
        self, producer: AIOKafkaProducer, data: Iterable, topic: str
    ):

        # To ensure we don't hit MessageSizeTooLargeErrors, split the data into chunks when sending
        # Add 10% to account for metadata sent in the message to be safe
        num_bytes = len(json.dumps(data).encode("utf-8")) * 1.10
        if num_bytes > producer._max_request_size:
            # Split into as many chunks as we need, limited by the length of `data`
            num_chunks = min(
                len(data), math.ceil(num_bytes / producer._max_request_size)
            )
            chunk_size = math.ceil(len(data) / num_chunks)

            logger.warning(
                f"Message size of {num_bytes} is larger than max_request_size {producer._max_request_size} - splitting message into {num_chunks} chunks of size {chunk_size}"
            )

            for chunk_start in range(0, len(data), chunk_size):
                await producer.send_and_wait(
                    topic, value=data[chunk_start : chunk_start + chunk_size]
                )

        # If the data is less than max_request_size, can send all at once
        else:
            await producer.send_and_wait(topic, value=data)
        logger.info(
            f"The number of records sent to topic:{topic}, record_no:{len(data)}"
        )

    async def get_data_batch(self, batch_size: Optional[int]) -> InternalDataFrame:
        batch = await self.consumer.getmany(
            timeout_ms=self.timeout_ms, max_records=batch_size
        )

        if len(batch) == 0:
            batch_data = []
        elif len(batch) > 1:
            logger.error(
                f"consumer should be subscribed to only one topic and partition, not {list(batch.keys())}"
            )
            batch_data = []
        else:
            for topic_partition, messages in batch.items():
                batch_data = [msg.value for msg in messages]

            logger.info(
                f"Received a batch with number_of_messages:{len(batch_data)} records from Kafka input_topic:{self.kafka_input_topic}"
            )
        return InternalDataFrame(batch_data)

    async def set_predictions(self, predictions: InternalDataFrame):
        predictions = [r.to_dict() for _, r in predictions.iterrows()]
        await self.message_sender(self.producer, predictions, self.kafka_output_topic)
