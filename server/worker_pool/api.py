# AI-MEMORY: FastAPI endpoints for worker pool architecture
# Benefits: 1) Fast response 2) No Celery task overhead 3) Direct Kafka publishing
# 4) Simple API design

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import uuid
import json
import logging
import math
from datetime import datetime

from .worker_processor import WorkMessage
from server.utils import Settings, ensure_topic_async

logger = logging.getLogger(__name__)

# New URL route for worker pool
router = APIRouter(prefix="/worker-pool", tags=["worker-pool"])

# Global Kafka producer for publishing work
_kafka_producer = None


async def get_kafka_producer():
    """Get or create the global Kafka producer"""
    global _kafka_producer
    if _kafka_producer is None:
        from aiokafka import AIOKafkaProducer

        settings = Settings()
        _kafka_producer = AIOKafkaProducer(
            **settings.kafka.to_kafka_kwargs(client_type="producer"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await _kafka_producer.start()
        logger.info("Kafka producer initialized")
    return _kafka_producer


class SubmitBatchRequest(BaseModel):
    records: List[Dict]
    skills: List[Dict]
    runtime_params: Dict
    batch_size: int = 1
    priority: int = 0
    api_key: Optional[str] = None  # LSE user's API key for result handling
    url: Optional[str] = None  # LSE URL to send predictions back to
    modelrun_id: Optional[int] = None  # Model run ID for LSE client creation


class SubmitBatchResponse(BaseModel):
    batch_id: str
    status: str
    message: str


@router.post("/submit-batch", response_model=SubmitBatchResponse)
async def submit_batch(request: SubmitBatchRequest) -> SubmitBatchResponse:
    """Submit batch for processing using worker pool"""

    try:
        # Generate unique batch ID
        batch_id = (
            f"batch_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Add modelrun_id to records if provided
        if request.modelrun_id:
            for record in request.records:
                record["modelrun_id"] = request.modelrun_id

        # Ensure topic exists
        await ensure_topic_async("worker_pool_input", num_partitions=50)

        # Get producer for chunking calculations
        producer = await get_kafka_producer()

        # Calculate message size with all records
        work_message_with_records = WorkMessage(
            batch_id=batch_id,
            skills=request.skills,
            runtime_params=request.runtime_params,
            input_topic="worker_pool_input",
            records=request.records,
            api_key=request.api_key,
            url=request.url,
            priority=request.priority,
        )

        # Calculate total message size with 10% buffer for metadata
        total_message_size = (
            len(json.dumps(work_message_with_records.__dict__).encode("utf-8")) * 1.10
        )

        # Check if we need to chunk the message
        if total_message_size > producer._max_request_size and len(request.records) > 1:
            # Calculate how many chunks we need
            num_chunks = min(
                len(request.records),
                math.ceil(total_message_size / producer._max_request_size),
            )
            chunk_size = math.ceil(len(request.records) / num_chunks)

            logger.warning(
                f"Message size of {total_message_size} is larger than max_request_size {producer._max_request_size} - "
                f"splitting {len(request.records)} records into {num_chunks} chunks of size {chunk_size}"
            )

            # Send chunked work messages
            for chunk_idx in range(0, len(request.records), chunk_size):
                chunk_records = request.records[chunk_idx : chunk_idx + chunk_size]

                # Create work message for this chunk
                chunked_work_message = WorkMessage(
                    batch_id=f"{batch_id}_chunk_{chunk_idx // chunk_size + 1}",
                    skills=request.skills,
                    runtime_params=request.runtime_params,
                    input_topic="worker_pool_input",
                    records=chunk_records,
                    api_key=request.api_key,
                    url=request.url,
                    priority=request.priority,
                )

                await producer.send_and_wait(
                    "worker_pool_input", value=chunked_work_message.__dict__
                )

            logger.info(
                f"Submitted batch {batch_id} to worker pool in {num_chunks} chunks"
            )
        else:
            # Send single work message
            await producer.send_and_wait(
                "worker_pool_input", value=work_message_with_records.__dict__
            )
            logger.info(f"Submitted batch {batch_id} to worker pool")

        return SubmitBatchResponse(
            batch_id=batch_id,
            status="submitted",
            message=f"Batch {batch_id} submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit batch: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit batch: {str(e)}")


async def cleanup_kafka_producer():
    """Cleanup the global Kafka producer"""
    global _kafka_producer
    if _kafka_producer:
        try:
            await _kafka_producer.stop()
            logger.info("Kafka producer stopped")
        except Exception as e:
            logger.warning(f"Error stopping Kafka producer: {e}")
        finally:
            _kafka_producer = None
