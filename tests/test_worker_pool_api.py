import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import logging
from server.worker_pool.api import SubmitBatchRequest, get_kafka_producer
from server.worker_pool.worker_processor import WorkMessage


@pytest.fixture
def large_batch_request():
    """Create a large batch request that should trigger chunking"""
    # Create a large record that when combined with metadata will exceed typical max_request_size
    large_text = "x" * 10000  # 10KB of text per record

    records = []
    for i in range(50):  # 50 records of 10KB each = ~500KB total
        records.append(
            {
                "text": large_text,
                "id": i,
                "metadata": {"large_field": "y" * 1000},  # Additional 1KB per record
            }
        )

    return SubmitBatchRequest(
        records=records,
        skills=[
            {
                "type": "ClassificationSkill",
                "name": "test_classifier",
                "instructions": "Test classification",
                "input_template": "{text}",
                "output_template": "{output}",
                "labels": ["label1", "label2"],
            }
        ],
        runtime_params={
            "type": "AsyncLiteLLMChatRuntime",
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "max_tokens": 100,
            "temperature": 0.5,
        },
        batch_size=10,
        priority=1,
        api_key="test-api-key",
        url="https://test.example.com",
        modelrun_id=12345,
    )


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer with small max_request_size to trigger chunking"""
    producer = AsyncMock()
    # Set small max_request_size to force chunking with our test data
    producer._max_request_size = 50000  # 50KB - smaller than our test data
    producer.send_and_wait = AsyncMock()
    return producer


@pytest.mark.asyncio
async def test_submit_batch_chunking_warning(
    large_batch_request, mock_kafka_producer, caplog
):
    """Test that chunking logic triggers warning message and creates multiple chunks"""

    with (
        patch(
            "server.worker_pool.api.get_kafka_producer",
            return_value=mock_kafka_producer,
        ),
        patch(
            "server.worker_pool.api.ensure_topic_async", new_callable=AsyncMock
        ) as mock_ensure_topic,
        caplog.at_level(logging.WARNING),
    ):

        from server.worker_pool.api import submit_batch

        # Call the submit_batch function
        response = await submit_batch(large_batch_request)

        # Verify successful response
        assert response.status == "submitted"
        assert "batch_" in response.batch_id

        # Verify ensure_topic_async was called
        mock_ensure_topic.assert_called_once_with(
            "worker_pool_input", num_partitions=50
        )

        # Check that warning message about chunking was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        chunking_warnings = [
            msg for msg in warning_messages if "splitting" in msg and "chunks" in msg
        ]

        assert (
            len(chunking_warnings) > 0
        ), f"Expected chunking warning, but got warnings: {warning_messages}"
        warning_msg = chunking_warnings[0]
        assert "larger than max_request_size" in warning_msg
        assert "splitting" in warning_msg
        assert "records into" in warning_msg
        assert "chunks" in warning_msg

        # Verify that multiple send_and_wait calls were made (chunking occurred)
        send_calls = mock_kafka_producer.send_and_wait.call_args_list
        assert (
            len(send_calls) > 1
        ), f"Expected multiple chunks, but only {len(send_calls)} calls made"

        # Verify all calls were to the correct topic
        for call in send_calls:
            args, kwargs = call
            assert args[0] == "worker_pool_input"

        # Verify that chunk batch_ids are correct
        chunk_data_list = []
        for call in send_calls:
            args, kwargs = call
            chunk_data = kwargs["value"]
            chunk_data_list.append(chunk_data)

            # Verify chunk has the right structure
            assert "batch_id" in chunk_data
            assert "_chunk_" in chunk_data["batch_id"]
            assert "records" in chunk_data
            assert "skills" in chunk_data
            assert "runtime_params" in chunk_data
            assert "api_key" in chunk_data
            assert "url" in chunk_data
            assert chunk_data["api_key"] == "test-api-key"
            assert chunk_data["url"] == "https://test.example.com"

        # Verify all records are present across chunks
        total_records_sent = sum(len(chunk["records"]) for chunk in chunk_data_list)
        assert total_records_sent == len(large_batch_request.records)

        # Verify modelrun_id was added to records
        for chunk_data in chunk_data_list:
            for record in chunk_data["records"]:
                assert record["modelrun_id"] == 12345


@pytest.mark.asyncio
async def test_submit_batch_no_chunking_small_data(caplog):
    """Test that small batches don't trigger chunking"""

    # Create small batch that won't trigger chunking
    small_request = SubmitBatchRequest(
        records=[{"text": "small", "id": 1}],
        skills=[{"type": "ClassificationSkill", "name": "test"}],
        runtime_params={"model": "test"},
        batch_size=1,
    )

    mock_producer = AsyncMock()
    mock_producer._max_request_size = 1000000  # 1MB - much larger than our small data
    mock_producer.send_and_wait = AsyncMock()

    with (
        patch("server.worker_pool.api.get_kafka_producer", return_value=mock_producer),
        patch("server.worker_pool.api.ensure_topic_async", new_callable=AsyncMock),
        caplog.at_level(logging.WARNING),
    ):

        from server.worker_pool.api import submit_batch

        response = await submit_batch(small_request)

        # Verify successful response
        assert response.status == "submitted"

        # Verify no chunking warning was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        chunking_warnings = [msg for msg in warning_messages if "splitting" in msg]
        assert len(chunking_warnings) == 0

        # Verify only one send call was made (no chunking)
        assert mock_producer.send_and_wait.call_count == 1


@pytest.mark.asyncio
async def test_submit_batch_preserves_all_fields_in_chunks(
    large_batch_request, mock_kafka_producer
):
    """Test that all WorkMessage fields are preserved when chunking"""

    with (
        patch(
            "server.worker_pool.api.get_kafka_producer",
            return_value=mock_kafka_producer,
        ),
        patch("server.worker_pool.api.ensure_topic_async", new_callable=AsyncMock),
    ):

        from server.worker_pool.api import submit_batch

        response = await submit_batch(large_batch_request)

        # Get all the chunk data
        send_calls = mock_kafka_producer.send_and_wait.call_args_list

        for call in send_calls:
            args, kwargs = call
            chunk_data = kwargs["value"]

            # Verify all expected fields are present and correct
            assert chunk_data["skills"] == large_batch_request.skills
            assert chunk_data["runtime_params"] == large_batch_request.runtime_params
            assert chunk_data["input_topic"] == "worker_pool_input"
            assert chunk_data["api_key"] == large_batch_request.api_key
            assert chunk_data["url"] == large_batch_request.url
            assert chunk_data["priority"] == large_batch_request.priority

            # Verify records have modelrun_id added
            for record in chunk_data["records"]:
                assert "modelrun_id" in record
                assert record["modelrun_id"] == large_batch_request.modelrun_id
