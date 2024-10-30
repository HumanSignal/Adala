import logging
import abc
import boto3
import json
import asyncio
import aiohttp
from csv import DictReader, DictWriter
from typing import Dict, Union, List, Optional, Iterable
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
        kafka_bootstrap_servers (Union[str, List[str]]): The Kafka bootstrap servers.
        kafka_input_topic (str): The Kafka input topic.
        kafka_output_topic (str): The Kafka output topic.
        timeout_ms (int): The timeout for the Kafka consumer.
    """

    # these are mandatory, but should be set by server
    kafka_bootstrap_servers: Optional[Union[str, List[str]]] = None
    kafka_input_topic: Optional[str] = None
    kafka_output_topic: Optional[str] = None
    timeout_ms: Optional[int] = None

    # these are set in initialize()
    consumer: Optional[AIOKafkaConsumer] = None
    producer: Optional[AIOKafkaProducer] = None

    async def initialize(self):
        assert (
            self.kafka_bootstrap_servers is not None
        ), "missing initialization for kafka_bootstrap_servers"
        assert (
            self.kafka_input_topic is not None
        ), "missing initialization for kafka_input_topic"
        assert (
            self.kafka_output_topic is not None
        ), "missing initialization for kafka_output_topic"
        assert self.timeout_ms is not None, "missing initialization for timeout_ms"

        self.consumer = AIOKafkaConsumer(
            self.kafka_input_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            group_id=self.kafka_input_topic,
        )
        await self.consumer.start()

        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks='all'
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
        record_no = 0
        try:
            for record in data:
                await producer.send_and_wait(topic, value=record)
                record_no += 1
                # print_text(f"Sent message: {record} to {topic=}")
            logger.info(
                f"The number of records sent to topic:{topic}, record_no:{record_no}"
            )
        finally:
            pass
            # print_text(f"No more messages for {topic=}")

    async def get_data_batch(self, batch_size: Optional[int]) -> InternalDataFrame:
        batch = await self.consumer.getmany(
            timeout_ms=self.timeout_ms, max_records=batch_size
        )
        await self.consumer.commit()

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
        predictions_iter = (r.to_dict() for _, r in predictions.iterrows())
        await self.message_sender(
            self.producer, predictions_iter, self.kafka_output_topic
        )
