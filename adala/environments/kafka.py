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
    """

    kafka_bootstrap_servers: Union[str, List[str]]
    kafka_input_topic: str
    kafka_output_topic: str

    async def initialize(self):
        # claim kafka topic from shared pool here?
        pass

    async def finalize(self):
        # release kafka topic to shared pool here?
        pass

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

    async def message_receiver(self, consumer: AIOKafkaConsumer, timeout: int = 3):
        await consumer.start()
        try:
            while True:
                try:
                    # Wait for the next message with a timeout
                    msg = await asyncio.wait_for(consumer.getone(), timeout=timeout)
                    # print_text(f"Received message: {msg.value}")
                    yield msg.value
                except asyncio.TimeoutError:
                    # print_text(
                        # f"No message received within the timeout {timeout} seconds"
                    # )
                    break
        finally:
            await consumer.stop()

    async def message_sender(
        self, producer: AIOKafkaProducer, data: Iterable, topic: str
    ):
        await producer.start()
        try:
            for record in data:
                await producer.send_and_wait(topic, value=record)
                # print_text(f"Sent message: {record} to {topic=}")
        finally:
            await producer.stop()
            # print_text(f"No more messages for {topic=}")

    async def get_next_batch(self, data_iterator, batch_size: int) -> List[Dict]:
        batch = []
        try:
            for _ in range(batch_size):
                data = await anext(data_iterator, None)
                if data is None:  # This checks if the iterator is exhausted
                    break
                batch.append(data)
        except StopAsyncIteration:
            pass
        return batch

    async def get_data_batch(self, batch_size: Optional[int]) -> InternalDataFrame:
        consumer = AIOKafkaConsumer(
            self.kafka_input_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            group_id="adala-consumer-group",  # TODO: make it configurable based on the environment
        )

        data_stream = self.message_receiver(consumer)
        batch = await self.get_next_batch(data_stream, batch_size)
        logger.info(
            f"Received a batch of {len(batch)} records from Kafka topic {self.kafka_input_topic}"
        )
        return InternalDataFrame(batch)

    async def set_predictions(self, predictions: InternalDataFrame):
        producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        predictions_iter = (r.to_dict() for _, r in predictions.iterrows())
        await self.message_sender(producer, predictions_iter, self.kafka_output_topic)


class FileStreamAsyncKafkaEnvironment(AsyncKafkaEnvironment):
    """
    Represents an asynchronous Kafka environment with file stream:
    - agent can retrieve data batch by batch from the input topic
    - agent can return its predictions to the output topic
    - input data is read from `input_file`
    - output data is stored to the `output_file`
    - errors are saved to the `error_file`
    """

    input_file: str
    output_file: str
    error_file: str
    pass_through_columns: Optional[List[str]] = None

    def _iter_csv_local(self, csv_file_path):
        """
        Read data from the CSV file and push it to the kafka topic.
        """

        with open(csv_file_path, "r") as csv_file:
            csv_reader = DictReader(csv_file)
            for row in csv_reader:
                yield row

    def _iter_csv_s3(self, s3_uri):
        """
        Read data from the CSV file in S3 and push it to the kafka topic.
        """
        # Assuming s3_uri format is "s3://bucket-name/path/to/file.csv"
        bucket_name, key = s3_uri.replace("s3://", "").split("/", 1)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        data = obj["Body"].read().decode("utf-8")
        csv_reader = DictReader(StringIO(data))
        for row in csv_reader:
            yield row

    async def initialize(self):
        """
        Initialize the environment: read data from the input file and push it to the kafka topic.
        """

        # TODO: Add support for other file types except CSV, and also for other cloud storage services
        if self.input_file.startswith("s3://"):
            csv_reader = self._iter_csv_s3(self.input_file)
        else:
            csv_reader = self._iter_csv_local(self.input_file)

        producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        await self.message_sender(producer, csv_reader, self.kafka_input_topic)

    async def finalize(self):
        """
        Finalize the environment: read data from the output kafka topic and write it to the output file.
        """

        consumer = AIOKafkaConsumer(
            self.kafka_output_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            group_id="consumer-group-output-topic",  # TODO: make it configurable based on the environment
        )

        data_stream = self.message_receiver(consumer)

        if self.output_file.startswith("s3://"):
            await self._write_to_s3(
                self.output_file,
                self.error_file,
                data_stream,
                self.pass_through_columns,
            )
        else:
            await self._write_to_local(
                self.output_file,
                self.error_file,
                data_stream,
                self.pass_through_columns,
            )

    async def _write_to_csv_fileobj(
        self, fileobj, error_fileobj, data_stream, column_names
    ):
        csv_writer, error_csv_writer = None, None
        error_columns = ["index", "message", "details"]
        while True:
            try:
                record = await anext(data_stream)
                if record.get("error") == True:
                    logger.error(f"Error occurred while processing record: {record}")
                    if error_csv_writer is None:
                        error_csv_writer = DictWriter(
                            error_fileobj, fieldnames=error_columns
                        )
                        error_csv_writer.writeheader()
                    error_csv_writer.writerow(
                        {k: record.get(k, "") for k in error_columns}
                    )
                else:
                    if csv_writer is None:
                        if column_names is None:
                            column_names = list(record.keys())
                        csv_writer = DictWriter(fileobj, fieldnames=column_names)
                        csv_writer.writeheader()
                    csv_writer.writerow({k: record.get(k, "") for k in column_names})
            except StopAsyncIteration:
                break

    async def _write_to_local(
        self, file_path: str, error_file_path: str, data_stream, column_names
    ):
        with open(file_path, "w") as csv_file, open(error_file_path, "w") as error_file:
            await self._write_to_csv_fileobj(
                csv_file, error_file, data_stream, column_names
            )

    async def _write_to_s3(
        self, s3_uri: str, s3_uri_errors: str, data_stream, column_names
    ):
        # Assuming s3_uri format is "s3://bucket-name/path/to/file.csv"
        bucket_name, key = s3_uri.replace("s3://", "").split("/", 1)
        error_bucket_name, error_key = s3_uri_errors.replace("s3://", "").split("/", 1)
        s3 = boto3.client("s3")
        with StringIO() as csv_file, StringIO() as error_file:
            await self._write_to_csv_fileobj(
                csv_file, error_file, data_stream, column_names
            )
            s3.put_object(Bucket=bucket_name, Key=key, Body=csv_file.getvalue())
            s3.put_object(
                Bucket=error_bucket_name, Key=error_key, Body=error_file.getvalue()
            )

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
