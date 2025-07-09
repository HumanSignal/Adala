# AI-MEMORY: WorkerProcessor class for per-worker processor architecture
# Benefits: 1) One processor per Celery worker 2) Shared Kafka queue 3) Automatic work distribution
# 4) Simple scaling 5) No complex pool management 6) Direct prediction queue communication

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import gc  # Added for memory tracking
import psutil  # Added for memory tracking
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from server.utils import (
    ensure_topic,
    ensure_topic_async,
    ensure_worker_pool_topics,
    ensure_worker_pool_input_topic,
)
from adala.utils.internal_data import InternalDataFrame
import weakref

logger = logging.getLogger(__name__)


# MEMORY TRACKING UTILITY
def _log_memory_usage(worker_id: str, stage: str):
    """Log current memory usage for debugging"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Worker {worker_id}: Memory at {stage}: {memory_mb:.1f}MB")
        return memory_mb
    except Exception as e:
        logger.debug(f"Worker {worker_id}: Error getting memory info: {e}")
        return 0


@dataclass
class WorkMessage:
    """Message format for work distribution"""

    batch_id: str
    skills: List[Dict]
    runtime_params: Dict
    input_topic: str
    records: List[Dict]  # The actual data to process
    api_key: Optional[str] = None  # LSE API key for this batch
    url: Optional[str] = None  # LSE URL to send predictions back to
    priority: int = 0

    @property
    def config_hash(self) -> str:
        """Generate hash for this configuration"""
        config_str = json.dumps(
            {
                "skills": sorted(self.skills, key=lambda x: x.get("name", "")),
                "runtime_params": self.runtime_params,
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_records_for_llm(self) -> List[Dict]:
        """Get records with api_key stripped out for LLM processing"""
        clean_records = []
        for record in self.records:
            # Create a copy without api_key
            clean_record = {k: v for k, v in record.items() if k != "api_key"}
            clean_records.append(clean_record)
        return clean_records


class WorkerProcessor:
    """Single processor that runs on each Celery worker"""

    def __init__(self, prediction_queue: Optional[asyncio.Queue] = None):
        self.worker_id = f"worker_{os.getpid()}"
        self.current_config_hash: Optional[str] = None
        self.current_batch_id: Optional[str] = (
            None  # Only store the current batch ID, not entire work message
        )
        self.current_runtime_config: Optional[Dict] = (
            None  # Store runtime config for recreation
        )
        self.environment = None
        self.skills = None
        self.runtime = None
        self.is_running = False
        self.last_processed_at = None
        self.config_switch_count = 0
        self.last_config_switch = None
        self.processed_batches = 0
        self.prediction_queue = prediction_queue  # Direct reference to the async queue
        self.restart_event = None  # Will be set by celery_integration.py

        # Memory monitoring
        self.last_memory_check = datetime.now()
        self.memory_check_interval = 30  # Check memory every 30 seconds when idle

        # Shared topics for input (output now goes directly to prediction queue)
        self.input_topic = "worker_pool_input"  # Shared topic for work messages

        # Log initial memory usage
        initial_memory = _log_memory_usage(self.worker_id, "process_starting")
        logger.info(f"Initialized worker processor: {self.worker_id}")

    def set_prediction_queue(self, prediction_queue: asyncio.Queue):
        """Set the prediction queue for direct communication"""
        self.prediction_queue = prediction_queue
        logger.info(f"Worker {self.worker_id}: Connected to prediction queue")

    async def initialize(self):
        """Initialize the processor"""
        from adala.environments import AsyncKafkaEnvironment
        from server.utils import Settings

        settings = Settings()

        # Add a random delay to prevent consumer group coordination storms
        # When multiple workers start simultaneously, they can overwhelm the Kafka coordinator
        delay = random.uniform(0, 20)  # Increased delay between 0-20 seconds
        logger.info(
            f"Worker {self.worker_id}: Starting initialization with {delay:.2f}s delay to prevent consumer group coordination storms"
        )
        await asyncio.sleep(delay)

        # Ensure input topic exists
        await ensure_worker_pool_input_topic()

        # Initialize environment for consuming work messages only (no output topic needed)
        self.environment = AsyncKafkaEnvironment(
            kafka_input_topic=self.input_topic,  # Listen to shared input topic
            kafka_output_topic="dummy_output_topic",  # Dummy topic - not used since we pass predictions directly
            timeout_ms=1000,  # 1 second timeout for work distribution
            kafka_kwargs={
                **settings.kafka.to_kafka_kwargs(client_type="consumer"),
                "group_id": "worker_pool_workers",  # Use same group ID for all workers for load balancing
            },
        )

        # Retry consumer group join with exponential backoff
        max_retries = 5
        base_delay = 1  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                await self.environment.initialize()
                logger.info(
                    f"Worker {self.worker_id}: Initialized successfully on attempt {attempt + 1}"
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Worker {self.worker_id}: Failed to initialize after {max_retries} attempts: {e}"
                    )
                    raise

                # Exponential backoff with jitter
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Worker {self.worker_id}: Initialization attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

    async def run_forever(self):
        """Main loop that runs forever on this worker"""
        self.is_running = True
        logger.info(f"Worker {self.worker_id}: Starting main loop")

        try:
            while self.is_running:
                logger.debug(f"Worker {self.worker_id}: Main loop iteration starting")

                # Check for new work and process it immediately
                await self._check_for_work()

                # Check memory periodically when idle
                logger.debug(
                    f"Worker {self.worker_id}: About to check memory threshold"
                )
                await self._check_memory_threshold()
                logger.debug(
                    f"Worker {self.worker_id}: Memory threshold check completed"
                )

                # Wait a bit before checking again
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id}: Main loop cancelled")
        except Exception as e:
            from server.worker_pool.celery_integration import RestartWorkerException

            if isinstance(e, RestartWorkerException):
                logger.info(
                    f"Worker {self.worker_id}: RestartWorkerException caught in main loop: {e}"
                )
                await self.cleanup()
                raise  # Re-raise restart exception to bubble up to celery
            else:
                logger.error(f"Worker {self.worker_id}: Error in main loop: {e}")
        finally:
            logger.info(f"Worker {self.worker_id}: Entering cleanup in finally block")
            await self.cleanup()
            logger.info(f"Worker {self.worker_id}: Cleanup completed in finally block")

    async def _check_for_work(self):
        """Check for new work from the shared queue and process it immediately"""
        try:
            # Get messages from work distribution topic with short timeout
            # One message contains one WorkMessage which contains multiple records based on
            # batch size sent from LSE
            data_batch = await asyncio.wait_for(
                self.environment.get_data_batch(batch_size=1), timeout=0.5
            )

            if not data_batch.empty:
                # Get the work message
                work_data = data_batch.iloc[0].to_dict()
                work_message = WorkMessage(**work_data)

                # Process this work immediately
                await self._assign_work(work_message)

        except asyncio.TimeoutError:
            # No work available, this is normal
            pass
        except Exception as e:
            from server.worker_pool.celery_integration import RestartWorkerException

            if isinstance(e, RestartWorkerException):
                raise  # Re-raise restart exception to bubble up to main loop
            else:
                logger.error(f"Worker {self.worker_id}: Error checking for work: {e}")

    async def _check_memory_threshold(self):
        """Check memory usage periodically and trigger restart if needed"""
        from server.worker_pool.celery_integration import RestartWorkerException

        try:
            # Only check memory every X seconds to avoid excessive logging
            now = datetime.now()
            time_since_last_check = (now - self.last_memory_check).total_seconds()

            if time_since_last_check < self.memory_check_interval:
                return

            # Update last check time
            self.last_memory_check = now

            # Get current memory usage
            current_memory = _log_memory_usage(self.worker_id, "periodic_memory_check")

            # Check if memory exceeds 340MB threshold
            if current_memory > 335:
                logger.warning(
                    f"Worker {self.worker_id}: Periodic memory check - {current_memory:.1f}MB exceeds 340MB threshold"
                )
                if hasattr(self, "restart_trigger") and self.restart_trigger:
                    logger.info(
                        f"Worker {self.worker_id}: About to trigger restart from periodic memory check"
                    )
                    await self.cleanup()
                    self.restart_trigger.trigger_restart(
                        f"Periodic memory check: {current_memory:.1f}MB exceeds 340MB threshold"
                    )
                    logger.error(
                        f"Worker {self.worker_id}: ERROR - This line should never be reached after trigger_restart!"
                    )
                else:
                    logger.error(
                        f"Worker {self.worker_id}: No restart trigger available for periodic memory check"
                    )

        except RestartWorkerException:
            # Re-raise restart exception to bubble up to main loop
            raise
        except Exception as e:
            logger.error(
                f"Worker {self.worker_id}: Error during periodic memory check: {e}"
            )

    async def _assign_work(self, work_message: WorkMessage):
        """Assign and process work immediately"""
        logger.info(f"Worker {self.worker_id}: Processing work {work_message.batch_id}")

        try:
            # Switch configuration if needed
            if self.current_config_hash != work_message.config_hash:
                await self._switch_configuration(work_message)

            # Update current work - store only essential identifiers, not entire work message
            self.current_batch_id = work_message.batch_id
            self.current_config_hash = work_message.config_hash

            # Process the work immediately
            await self._process_work(work_message)

            logger.info(
                f"Worker {self.worker_id}: Successfully processed work {work_message.batch_id}"
            )

        except Exception as e:
            from server.worker_pool.celery_integration import RestartWorkerException

            if isinstance(e, RestartWorkerException):
                raise  # Re-raise restart exception to bubble up to main loop
            else:
                logger.error(
                    f"Worker {self.worker_id}: Failed to process work {work_message.batch_id}: {e}"
                )

    async def _process_work(self, work_message: WorkMessage):
        """Process the assigned work"""
        if not self.skills or not self.runtime:
            return

        data_batch = None
        predictions = None

        # MEMORY TRACKING: Log memory at start of processing
        start_memory = _log_memory_usage(self.worker_id, "start_processing")

        try:
            # Process the data directly from the work message
            if hasattr(work_message, "records") and work_message.records:
                data_batch = InternalDataFrame(work_message.get_records_for_llm())
                _log_memory_usage(self.worker_id, "after_dataframe_creation")
            else:
                logger.warning(
                    f"Worker {self.worker_id}: No records found in work message {work_message.batch_id}"
                )
                return

            # Process the batch
            before_llm_memory = _log_memory_usage(
                self.worker_id, "before_llm_processing"
            )

            predictions = await self.skills.aapply(data_batch, runtime=self.runtime)

            after_llm_memory = _log_memory_usage(self.worker_id, "after_llm_processing")
            llm_memory_diff = after_llm_memory - before_llm_memory
            if llm_memory_diff > 10:  # More than 10MB growth
                logger.warning(
                    f"Worker {self.worker_id}: LLM processing increased memory by {llm_memory_diff:.1f}MB"
                )

            # Send predictions directly to prediction queue instead of Kafka
            if self.prediction_queue:
                # Extract modelrun_id from first record for this batch
                modelrun_id = None
                if work_message.records:
                    modelrun_id = work_message.records[0].get("modelrun_id")

                await self._add_prediction_to_queue(
                    work_message.batch_id,
                    predictions,
                    work_message.api_key,
                    work_message.url,
                    modelrun_id,
                )
                _log_memory_usage(self.worker_id, "after_queue_add")
            else:
                logger.warning(
                    f"Worker {self.worker_id}: No prediction queue available, predictions for batch {work_message.batch_id} will be lost"
                )

            self.last_processed_at = datetime.now()
            self.processed_batches += 1

        except Exception as e:
            from server.worker_pool.celery_integration import RestartWorkerException

            if isinstance(e, RestartWorkerException):
                raise  # Re-raise restart exception to bubble up to main loop
            else:
                logger.error(f"Worker {self.worker_id}: Error processing work: {e}")
        finally:
            # MEMORY TRACKING: Log memory during cleanup
            cleanup_start_memory = _log_memory_usage(self.worker_id, "cleanup_start")

            # Basic cleanup
            if data_batch is not None:
                del data_batch
                data_batch = None
            if predictions is not None:
                del predictions
                predictions = None

            # Force garbage collection
            gc.collect()

            cleanup_end_memory = _log_memory_usage(self.worker_id, "cleanup_end")
            total_memory_diff = cleanup_end_memory - start_memory

            if total_memory_diff > 5:  # More than 5MB not recovered
                logger.warning(
                    f"Worker {self.worker_id}: Memory not fully recovered - {total_memory_diff:.1f}MB still allocated after cleanup"
                )
            else:
                logger.info(
                    f"Worker {self.worker_id}: Memory cleanup successful - {total_memory_diff:.1f}MB change"
                )

            # Check if memory usage exceeds 340MB and trigger restart if needed
            if cleanup_end_memory > 340:
                logger.warning(
                    f"Worker {self.worker_id}: Memory usage {cleanup_end_memory:.1f}MB exceeds 340MB threshold"
                )
                if hasattr(self, "restart_trigger") and self.restart_trigger:
                    try:
                        self.restart_trigger.trigger_restart(
                            f"Memory usage {cleanup_end_memory:.1f}MB exceeds 340MB threshold"
                        )
                    except Exception as restart_exc:
                        # Re-raise restart exception to bubble up to main loop
                        from server.worker_pool.celery_integration import (
                            RestartWorkerException,
                        )

                        if isinstance(restart_exc, RestartWorkerException):
                            raise
                        else:
                            logger.error(
                                f"Worker {self.worker_id}: Unexpected error during restart trigger: {restart_exc}"
                            )
                            raise
                else:
                    logger.error(
                        f"Worker {self.worker_id}: No restart trigger available for memory threshold restart"
                    )

    async def _add_prediction_to_queue(
        self,
        batch_id: str,
        predictions,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        modelrun_id: Optional[str] = None,
    ):
        """Add predictions to the processing queue with api_key for LSE client"""
        try:
            prediction_data = {
                "batch_id": batch_id,
                "predictions": predictions,
                "api_key": api_key,  # Include api_key for LSE client creation
                "url": url,  # Include URL for LSE client creation
                "modelrun_id": modelrun_id,  # Include modelrun_id for LSE client creation
                "timestamp": datetime.now(),
            }

            # Use put_nowait to avoid blocking the worker processor
            self.prediction_queue.put_nowait(prediction_data)

        except Exception as e:
            logger.error(
                f"Worker {self.worker_id}: Error adding prediction to queue: {e}"
            )

    async def _switch_configuration(self, work_message: WorkMessage):
        """
        Switch to new configuration for the worker
        This is done when the worker receives a new work message with different skills or runtime
        than the last message.
        """
        logger.info(f"Worker {self.worker_id}: Switching configuration")

        # Store runtime config for recreation after each batch
        self.current_runtime_config = work_message.runtime_params.copy()

        # Create new skills and runtime
        self.skills = self._create_skills_from_config(work_message.skills)
        self.runtime = self._create_runtime_from_config(work_message.runtime_params)

        self.config_switch_count += 1
        self.last_config_switch = datetime.now()

    def _create_skills_from_config(self, skills_config: List[Dict]):
        """Create skills from configuration"""
        from adala.skills import (
            LinearSkillSet,
            TransformSkill,
            ClassificationSkill,
            EntityExtraction,
            LabelStudioSkill,
        )

        skill_instances = []
        for skill_config in skills_config:
            skill_type = skill_config.get("type")
            if skill_type == "TransformSkill":
                skill = TransformSkill(**skill_config)
            elif skill_type == "ClassificationSkill":
                skill = ClassificationSkill(**skill_config)
            elif skill_type == "EntityExtraction":
                skill = EntityExtraction(**skill_config)
            elif skill_type == "LabelStudioSkill":
                skill = LabelStudioSkill(**skill_config)
            else:
                logger.warning(f"Unknown skill type: {skill_type}, skipping")
                continue
            skill_instances.append(skill)

        return LinearSkillSet(skills=skill_instances)

    def _create_runtime_from_config(self, runtime_config: Dict):
        """Create runtime from configuration"""
        from adala.runtimes import (
            AsyncLiteLLMChatRuntime,
            AsyncLiteLLMVisionRuntime,
            AsyncOpenAIChatRuntime,
            AsyncOpenAIVisionRuntime,
        )

        runtime_type = runtime_config.get("type", "AsyncLiteLLMChatRuntime")

        if runtime_type == "AsyncLiteLLMChatRuntime":
            return AsyncLiteLLMChatRuntime(**runtime_config)
        elif runtime_type == "AsyncLiteLLMVisionRuntime":
            return AsyncLiteLLMVisionRuntime(**runtime_config)
        elif runtime_type == "AsyncOpenAIChatRuntime":
            return AsyncOpenAIChatRuntime(**runtime_config)
        elif runtime_type == "AsyncOpenAIVisionRuntime":
            return AsyncOpenAIVisionRuntime(**runtime_config)
        else:
            # Default to AsyncLiteLLMChatRuntime if type is not recognized
            logger.warning(
                f"Unknown runtime type: {runtime_type}, using AsyncLiteLLMChatRuntime"
            )
            return AsyncLiteLLMChatRuntime(**runtime_config)

    async def cleanup(self):
        """Clean up resources"""
        self.is_running = False

        if self.environment:
            try:
                await self.environment.finalize()
                logger.info(f"Worker {self.worker_id}: Environment finalized")
            except Exception as e:
                logger.warning(
                    f"Worker {self.worker_id}: Error finalizing environment: {e}"
                )
            finally:
                self.environment = None

        # Clear references
        self.skills = None
        self.runtime = None
        self.current_batch_id = None
        self.current_config_hash = None
        self.current_runtime_config = None

        logger.info(f"Worker {self.worker_id}: Cleaned up")

    def get_status(self) -> Dict:
        """Get status of this worker"""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "current_batch_id": self.current_batch_id,
            "current_config_hash": self.current_config_hash,
            "config_switch_count": self.config_switch_count,
            "last_processed_at": self.last_processed_at,
            "processed_batches": self.processed_batches,
            "last_config_switch": self.last_config_switch,
        }
