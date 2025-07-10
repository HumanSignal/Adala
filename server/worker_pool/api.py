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

        # Create work message with API key, URL, and modelrun_id
        work_message = WorkMessage(
            batch_id=batch_id,
            skills=request.skills,
            runtime_params=request.runtime_params,
            input_topic="worker_pool_input",
            records=request.records,
            api_key=request.api_key,  # Pass through the API key
            url=request.url,  # Pass through the URL
            priority=request.priority,
        )

        # Add modelrun_id to each record for processing
        if request.modelrun_id:
            for record in work_message.records:
                record["modelrun_id"] = request.modelrun_id

        # Ensure topic exists
        await ensure_topic_async("worker_pool_input", num_partitions=50)

        # Get producer and publish work message
        producer = await get_kafka_producer()
        await producer.send_and_wait("worker_pool_input", value=work_message.__dict__)

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
