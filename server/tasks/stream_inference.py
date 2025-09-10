import asyncio
import json
import os
import psutil
import time
import traceback

from adala.agents import Agent

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import UnknownTopicOrPartitionError
from celery import Celery
from celery.signals import worker_process_shutdown, worker_process_init
from server.handlers.result_handlers import ResultHandler
from server.utils import (
    Settings,
    delete_topic,
    ensure_topic,
    get_input_topic_name,
    get_output_topic_name,
    init_logger,
)

import server.worker_pool.celery_integration

logger = init_logger(__name__)

settings = Settings()

app = Celery(
    "worker",
    broker=settings.redis.to_url(),
    backend=settings.redis.to_url(),
    accept_content=["json", "pickle"],
    broker_connection_retry_on_startup=True,
    worker_max_memory_per_child=settings.celery_worker_max_memory_per_child_kb,
    **{f"redis_{k}": v for k, v in settings.redis.to_kwargs().items()},
)


@worker_process_init.connect
def worker_process_init_handler(**kwargs):
    """Called when a worker process starts."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(
        f"Worker process starting. PID: {os.getpid()}, "
        f"Memory RSS: {mem_info.rss / 1024 / 1024:.2f}MB"
    )


@worker_process_shutdown.connect
def worker_process_shutdown_handler(**kwargs):
    """Called when a worker process shuts down."""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(
        f"Worker process shutting down. PID: {os.getpid()}, "
        f"Memory RSS: {mem_info.rss / 1024 / 1024:.2f}MB"
    )
