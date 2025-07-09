# AI-MEMORY: OutputProcessor class that receives predictions via shared async queue
# Benefits: 1) No Kafka needed for output 2) Clean separation of concerns 3) Integrates with LSE per-batch API keys
# 4) Runs alongside WorkerProcessor with shared async queue communication 5) Efficient LSE client caching per API key

import asyncio
import hashlib
import logging
import os
import time
import gc  # Added for memory tracking
import psutil  # Added for memory tracking
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from server.handlers.result_handlers import ResultHandler, LSEHandler

logger = logging.getLogger(__name__)


# MEMORY TRACKING UTILITY
def _log_memory_usage(processor_id: str, stage: str):
    """Log current memory usage for debugging"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(
            f"Output processor {processor_id}: Memory at {stage}: {memory_mb:.1f}MB"
        )
        return memory_mb
    except Exception as e:
        logger.debug(f"Output processor {processor_id}: Error getting memory info: {e}")
        return 0


def _mask_api_key(api_key: str) -> str:
    """Safely mask API key for logging, showing only first 4 and last 4 characters"""
    if not api_key or len(api_key) < 8:
        return "***masked***"
    return f"{api_key[:4]}...{api_key[-4:]}"


class LSEClientCache:
    """Cache for LSE clients by API key with expiration"""

    def __init__(self, expiration_hours: int = 1):
        self.clients: Dict[str, Dict[str, Any]] = (
            {}
        )  # api_key_hash -> {client, last_used, modelrun_id}
        self.expiration_hours = expiration_hours
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()

    def _hash_api_key(self, api_key: str, modelrun_id: int = None) -> str:
        """Generate hash for API key and modelrun_id combination"""
        key_data = f"{api_key}:{modelrun_id}" if modelrun_id else api_key
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    async def _create_client(
        self, cache_key: str, api_key: str, url: str, modelrun_id: int
    ) -> LSEHandler:
        """Create a new LSE client"""
        if not url:
            logger.error(
                f"No URL provided for LSE client creation (modelrun_id: {modelrun_id})"
            )
            return None

        client = LSEHandler(api_key=api_key, url=url, modelrun_id=modelrun_id)

        # Cache the client
        self.clients[cache_key] = {
            "client": client,
            "last_used": datetime.now(),
            "modelrun_id": modelrun_id,  # Store for debugging
        }

        logger.info(
            f"Created new LSE client for cache key {cache_key} (modelrun_id: {modelrun_id}, api_key: {_mask_api_key(api_key)})"
        )
        return client

    async def get_client(
        self, api_key: str, url: str = None, modelrun_id: int = None
    ) -> Optional[LSEHandler]:
        """Get or create LSE client for the given API key and modelrun_id combination"""
        if not api_key:
            return None

        if not modelrun_id:
            logger.error("No modelrun_id provided - cannot create LSE client")
            return None

        # Clean up expired clients periodically
        await self._cleanup_expired_clients()

        cache_key = self._hash_api_key(api_key, modelrun_id)

        # Check if we have a cached client
        if cache_key in self.clients:
            cached_client = self.clients[cache_key]
            # If the client is no longer ready, recreate it - this fallback is unlikely to happen
            if not cached_client["client"].ready():
                await self._create_client(cache_key, api_key, url, modelrun_id)
            cached_client["last_used"] = datetime.now()
            logger.info(
                f"Using cached LSE client for cache key {cache_key} (modelrun_id: {modelrun_id}, api_key: {_mask_api_key(api_key)})"
            )
            return cached_client["client"]

        # Create new client
        try:
            client = await self._create_client(cache_key, api_key, url, modelrun_id)
            return client

        except Exception as e:
            logger.error(
                f"Failed to create LSE client for cache key {cache_key} (modelrun_id: {modelrun_id}, api_key: {_mask_api_key(api_key)}): {e}"
            )
            return None

    async def _cleanup_expired_clients(self):
        """Remove expired clients from cache"""
        now = datetime.now()

        # Only cleanup every cleanup_interval seconds
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return

        self.last_cleanup = now
        expiration_time = now - timedelta(hours=self.expiration_hours)

        expired_keys = []
        for api_key_hash, client_info in self.clients.items():
            if client_info["last_used"] < expiration_time:
                expired_keys.append(api_key_hash)

        for api_key_hash in expired_keys:
            del self.clients[api_key_hash]
            logger.info(f"Removed expired LSE client for cache key {api_key_hash}")

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired LSE clients")


class OutputProcessor:
    """Processor that receives predictions from WorkerProcessor via shared async queue and sends to LSE"""

    def __init__(self, prediction_queue: Optional[asyncio.Queue] = None):
        self.processor_id = f"output_processor_{os.getpid()}"
        self.is_running = False
        self.lse_client_cache = LSEClientCache(expiration_hours=1)
        self.processed_batches = 0
        self.last_processed_at = None
        self.prediction_queue = prediction_queue
        self.restart_trigger = None  # Will be set by celery_integration.py
        self.restart_event = None  # Will be set by celery_integration.py
        # Log initial memory usage
        initial_memory = _log_memory_usage(self.processor_id, "process_starting")
        logger.info(f"Initialized output processor: {self.processor_id}")

    async def initialize(self):
        """Initialize the output processor"""
        logger.info(f"Output processor {self.processor_id}: Initialized successfully")

    async def run_forever(self):
        """Main loop that processes predictions from the async queue"""
        self.is_running = True
        logger.info(f"Output processor {self.processor_id}: Starting main loop")

        try:
            while self.is_running:
                # Check if restart has been triggered by other processor
                if self.restart_event and self.restart_event.is_set():
                    logger.info(f"Output processor {self.processor_id}: Restart event detected - exiting gracefully")
                    from server.worker_pool.celery_integration import RestartWorkerException
                    raise RestartWorkerException("Coordinated restart from WorkerProcessor")
                
                try:
                    # Wait for predictions from the queue with timeout
                    prediction_data = await asyncio.wait_for(
                        self.prediction_queue.get(),
                        timeout=1.0,
                    )

                    # Process the prediction
                    await self._process_prediction(prediction_data)

                    # Mark task as done
                    self.prediction_queue.task_done()

                except asyncio.TimeoutError:
                    # No predictions available, continue loop (also check restart event on next iteration)
                    continue
                except Exception as e:
                    from server.worker_pool.celery_integration import RestartWorkerException
                    if isinstance(e, RestartWorkerException):
                        await self.cleanup()
                        raise  # Re-raise restart exception to bubble up to main loop
                    else:
                        logger.error(
                            f"Output processor {self.processor_id}: Error processing prediction: {e}"
                        )

        except asyncio.CancelledError:
            logger.info(f"Output processor {self.processor_id}: Main loop cancelled")
        except Exception as e:
            from server.worker_pool.celery_integration import RestartWorkerException
            if isinstance(e, RestartWorkerException):
                logger.info(f"Output processor {self.processor_id}: RestartWorkerException caught in main loop: {e}")
                await self.cleanup()
                raise  # Re-raise restart exception to bubble up to celery
            else:
                logger.error(
                    f"Output processor {self.processor_id}: Error in main loop: {e}"
                )
        finally:
            logger.info(f"Output processor {self.processor_id}: Entering cleanup in finally block")
            await self.cleanup()
            logger.info(f"Output processor {self.processor_id}: Cleanup completed in finally block")

    async def _process_prediction(self, prediction_data: Dict[str, Any]):
        """Process a single prediction batch with per-batch API key handling"""
        records = None
        lse_client = None

        # MEMORY TRACKING: Log memory at start of processing
        start_memory = _log_memory_usage(self.processor_id, "start_processing")

        try:
            batch_id = prediction_data.get("batch_id", "unknown")
            predictions = prediction_data.get("predictions")
            api_key = prediction_data.get("api_key")
            url = prediction_data.get("url")
            modelrun_id = prediction_data.get("modelrun_id")

            # Check if we have an API key
            if not api_key:
                logger.error(
                    f"Output processor {self.processor_id}: No API key provided for batch {batch_id} - cannot send to LSE"
                )
                return

            # Check if we have a URL
            if not url:
                logger.error(
                    f"Output processor {self.processor_id}: No URL provided for batch {batch_id} - cannot send to LSE"
                )
                return

            # Check if we have a modelrun_id
            if not modelrun_id:
                logger.error(
                    f"Output processor {self.processor_id}: No modelrun_id provided for batch {batch_id} - cannot send to LSE"
                )
                return

            # Check if predictions is None or empty DataFrame
            if predictions is None:
                logger.warning(
                    f"Output processor {self.processor_id}: No predictions in batch {batch_id}"
                )
                return
            if hasattr(predictions, "empty") and predictions.empty:
                logger.warning(
                    f"Output processor {self.processor_id}: Empty predictions DataFrame in batch {batch_id}"
                )
                return

            logger.info(
                f"Output processor {self.processor_id}: Processing batch {batch_id} with predictions (modelrun_id: {modelrun_id})"
            )

            # Convert predictions to the format expected by result handlers
            before_conversion_memory = _log_memory_usage(
                self.processor_id, "before_conversion"
            )

            records = self._convert_predictions_to_records(predictions)

            after_conversion_memory = _log_memory_usage(
                self.processor_id, "after_conversion"
            )

            # Check if we got valid records after conversion
            if not records or len(records) == 0:
                logger.error(
                    f"Output processor {self.processor_id}: No valid records after conversion for batch {batch_id}"
                )
                return

            # Get LSE client for this API key with URL and modelrun_id
            before_client_memory = _log_memory_usage(
                self.processor_id, "before_lse_client"
            )

            lse_client = await self.lse_client_cache.get_client(
                api_key, url=url, modelrun_id=modelrun_id
            )

            after_client_memory = _log_memory_usage(
                self.processor_id, "after_lse_client"
            )

            if not lse_client:
                logger.error(
                    f"Output processor {self.processor_id}: Failed to get LSE client for batch {batch_id}"
                )
                return

            # Send results to LSE
            before_send_memory = _log_memory_usage(self.processor_id, "before_lse_send")

            await self._handle_results(lse_client, records)

            after_send_memory = _log_memory_usage(self.processor_id, "after_lse_send")

            self.processed_batches += 1
            self.last_processed_at = datetime.now()

            logger.info(
                f"Output processor {self.processor_id}: Successfully processed batch {batch_id}"
            )

        except Exception as e:
            from server.worker_pool.celery_integration import RestartWorkerException
            if isinstance(e, RestartWorkerException):
                raise  # Re-raise restart exception to bubble up to main loop
            else:
                logger.error(
                    f"Output processor {self.processor_id}: Error processing prediction: {e}"
                )
                logger.error(
                    f"Output processor {self.processor_id}: Prediction data type: {type(prediction_data.get('predictions', None))}"
                )
                if hasattr(prediction_data.get("predictions", None), "shape"):
                    logger.error(
                        f"Output processor {self.processor_id}: Prediction shape: {prediction_data['predictions'].shape}"
                    )
        finally:
            # MEMORY TRACKING: Log memory during cleanup
            cleanup_start_memory = _log_memory_usage(self.processor_id, "cleanup_start")

            # Basic cleanup
            if records is not None:
                del records
                records = None
            if prediction_data is not None:
                del prediction_data
                prediction_data = None
            if lse_client is not None:
                lse_client = None

            # Force garbage collection
            gc.collect()

            cleanup_end_memory = _log_memory_usage(self.processor_id, "cleanup_end")
            total_memory_diff = cleanup_end_memory - start_memory

            if total_memory_diff > 5:  # More than 5MB not recovered
                logger.warning(
                    f"Output processor {self.processor_id}: Memory not fully recovered - {total_memory_diff:.1f}MB still allocated after cleanup"
                )
            else:
                logger.info(
                    f"Output processor {self.processor_id}: Memory cleanup successful - {total_memory_diff:.1f}MB change"
                )

    def _convert_predictions_to_records(self, predictions) -> list:
        """Convert predictions to records format expected by result handlers"""
        try:
            # Predictions should always be a DataFrame from Adala skills
            records = predictions.to_dict("records")
            return records

        except Exception as e:
            logger.error(f"Error converting predictions to records: {e}")
            logger.error(f"Predictions type: {type(predictions)}")
            return []

    async def _handle_results(self, result_handler: ResultHandler, records: list):
        """Handle results using the result handler, with async support"""
        try:
            logger.info(
                f"Output processor {self.processor_id}: Sending {len(records)} records to LSE"
            )

            # Check if the handler is async-compatible
            if asyncio.iscoroutinefunction(result_handler.__call__):
                await result_handler(records)
                logger.info(
                    f"Output processor {self.processor_id}: Successfully sent records via async handler"
                )
            else:
                # Run sync handler in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, result_handler, records)
                logger.info(
                    f"Output processor {self.processor_id}: Successfully sent records via sync handler"
                )

        except Exception as e:
            logger.error(
                f"Output processor {self.processor_id}: Error handling results: {e}"
            )
            # Log the first few records for debugging
            if records:
                logger.error(
                    f"Output processor {self.processor_id}: Sample record: {records[0]}"
                )
            raise

    async def cleanup(self):
        """Clean up resources"""
        self.is_running = False

        # Process any remaining items in the queue
        try:
            while not self.prediction_queue.empty():
                try:
                    prediction_data = self.prediction_queue.get_nowait()
                    await self._process_prediction(prediction_data)
                    self.prediction_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(
                        f"Error processing remaining prediction during cleanup: {e}"
                    )
        except Exception as e:
            logger.error(f"Error during queue cleanup: {e}")

        logger.info(f"Output processor {self.processor_id}: Cleaned up")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the processor with detailed cache information"""
        cache_info = {
            "total_clients": len(self.lse_client_cache.clients),
            "client_ages": {},
            "expiration_hours": self.lse_client_cache.expiration_hours,
        }

        # Get cache age information - safely access the cache structure
        current_time = datetime.now()
        for cache_key, client_info in self.lse_client_cache.clients.items():
            last_used = client_info.get("last_used", current_time)
            age_hours = (current_time - last_used).total_seconds() / 3600
            # Only include the cache key (hash) - no sensitive information
            cache_info["client_ages"][cache_key] = round(age_hours, 2)

        status = {
            "processor_id": self.processor_id,
            "is_running": self.is_running,
            "processed_batches": self.processed_batches,
            "last_processed_at": (
                self.last_processed_at.isoformat() if self.last_processed_at else None
            ),
            "queue_size": self.prediction_queue.qsize() if self.prediction_queue else 0,
            "queue_maxsize": (
                self.prediction_queue.maxsize if self.prediction_queue else 0
            ),
            "queue_empty": (
                self.prediction_queue.empty() if self.prediction_queue else True
            ),
            "queue_full": (
                self.prediction_queue.full() if self.prediction_queue else False
            ),
            "lse_client_cache": cache_info,
        }

        return status
