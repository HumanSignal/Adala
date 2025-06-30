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
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from server.utils import ensure_topic, ensure_topic_async, ensure_worker_pool_topics, ensure_worker_pool_input_topic
from adala.utils.internal_data import InternalDataFrame
import weakref

logger = logging.getLogger(__name__)

@dataclass
class WorkMessage:
    """Message format for work distribution"""
    batch_id: str
    skills: List[Dict]
    runtime_params: Dict
    batch_size: int
    input_topic: str
    output_topic: str
    records: List[Dict]  # The actual data to process
    api_key: Optional[str] = None  # LSE API key for this batch
    url: Optional[str] = None  # LSE URL to send predictions back to
    priority: int = 0
    
    @property
    def config_hash(self) -> str:
        """Generate hash for this configuration"""
        config_str = json.dumps({
            'skills': sorted(self.skills, key=lambda x: x.get('name', '')),
            'runtime_params': self.runtime_params,
            'batch_size': self.batch_size
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_records_for_llm(self) -> List[Dict]:
        """Get records with api_key stripped out for LLM processing"""
        clean_records = []
        for record in self.records:
            # Create a copy without api_key
            clean_record = {k: v for k, v in record.items() if k != 'api_key'}
            clean_records.append(clean_record)
        return clean_records

class WorkerProcessor:
    """Single processor that runs on each Celery worker"""
    
    def __init__(self, prediction_queue: Optional[asyncio.Queue] = None):
        self.worker_id = f"worker_{os.getpid()}"
        self.current_config_hash: Optional[str] = None
        self.current_config: Optional[WorkMessage] = None
        self.environment = None
        self.skills = None
        self.runtime = None
        self.is_running = False
        self.last_processed_at = None
        self.config_switch_count = 0
        self.last_config_switch = None
        self.processed_batches = 0
        self.prediction_queue = prediction_queue  # Direct reference to the async queue
        
        # Shared topics for input (output now goes directly to prediction queue)
        self.input_topic = "worker_pool_input"  # Shared topic for work messages
        
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
        
        # Add a small random delay to help with partition distribution
        delay = random.uniform(0, 2)
        logger.info(f"Worker {self.worker_id}: Starting initialization with {delay:.2f}s delay")
        await asyncio.sleep(delay)
        
        # Debug: Log the Kafka settings being used
        logger.info(f"Worker {self.worker_id}: Kafka settings - bootstrap_servers: {settings.kafka.bootstrap_servers}")
        logger.info(f"Worker {self.worker_id}: Kafka kwargs: {settings.kafka.to_kafka_kwargs()}")
        
        # Ensure input topic exists with exactly 50 partitions (no need for output topic since we use direct communication)
        logger.info(f"Worker {self.worker_id}: Ensuring worker pool input topic exists with 50 partitions...")
        from server.utils import ensure_worker_pool_input_topic
        await ensure_worker_pool_input_topic()
        
        # Initialize environment for consuming work messages only (no output topic needed)
        self.environment = AsyncKafkaEnvironment(
            kafka_input_topic=self.input_topic,  # Listen to shared input topic
            kafka_output_topic="dummy_output_topic",  # Dummy topic - not used since we pass predictions directly
            timeout_ms=1000,  # 1 second timeout for work distribution
            kafka_kwargs={
                **settings.kafka.to_kafka_kwargs(),
                'group_id': 'worker_pool_workers',  # Use same group ID for all workers for load balancing
            }
        )
        await self.environment.initialize()
        
        # Debug: Check consumer assignment
        if hasattr(self.environment, 'consumer') and self.environment.consumer:
            assignment = self.environment.consumer.assignment()
            logger.info(f"Worker {self.worker_id}: Consumer assignment: {assignment}")
            
            # Get partition info
            partitions = self.environment.consumer.partitions_for_topic(self.input_topic)
            logger.info(f"Worker {self.worker_id}: Available partitions: {partitions}")
        
        logger.info(f"Worker {self.worker_id}: Initialized successfully")
    
    async def run_forever(self):
        """Main loop that runs forever on this worker"""
        self.is_running = True
        logger.info(f"Worker {self.worker_id}: Starting main loop")
        
        try:
            while self.is_running:
                # Check for new work and process it immediately
                await self._check_for_work()
                
                # Wait a bit before checking again
                await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id}: Main loop cancelled")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error in main loop: {e}")
        finally:
            await self.cleanup()
    
    async def _check_for_work(self):
        """Check for new work from the shared queue and process it immediately"""
        try:
            # Get messages from work distribution topic with short timeout
            # One message contains one WorkMessage which contains multiple records based on 
            # batch size sent from LSE
            data_batch = await asyncio.wait_for(
                self.environment.get_data_batch(batch_size=1),
                timeout=0.5
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
            logger.error(f"Worker {self.worker_id}: Error checking for work: {e}")
    
    async def _assign_work(self, work_message: WorkMessage):
        """Assign and process work immediately"""
        logger.info(f"Worker {self.worker_id}: Processing work {work_message.batch_id}")
        
        try:
            # Switch configuration if needed
            if self.current_config_hash != work_message.config_hash:
                await self._switch_configuration(work_message)
            
            # Update current work
            self.current_config = work_message
            self.current_config_hash = work_message.config_hash
            
            # Process the work immediately
            await self._process_work(work_message)
            
            logger.info(f"Worker {self.worker_id}: Successfully processed work {work_message.batch_id}")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to process work {work_message.batch_id}: {e}")
    
    async def _process_work(self, work_message: WorkMessage):
        """Process the assigned work"""
        if not self.skills or not self.runtime:
            return
        
        try:
            # Process the data directly from the work message
            if hasattr(work_message, 'records') and work_message.records:
                data_batch = InternalDataFrame(work_message.get_records_for_llm())
            else:
                logger.warning(f"Worker {self.worker_id}: No records found in work message {work_message.batch_id}")
                return
            
            # Process the batch
            predictions = await self.skills.aapply(data_batch, runtime=self.runtime)
            
            # Send predictions directly to prediction queue instead of Kafka
            if self.prediction_queue:
                await self._add_prediction_to_queue(work_message.batch_id, predictions, work_message.api_key, work_message.url)
                logger.debug(f"Worker {self.worker_id}: Sent predictions to queue for batch {work_message.batch_id}")
            else:
                logger.warning(f"Worker {self.worker_id}: No prediction queue available, predictions for batch {work_message.batch_id} will be lost")
            
            self.last_processed_at = datetime.now()
            self.processed_batches += 1
            
            logger.debug(f"Worker {self.worker_id}: Processed batch for {work_message.batch_id}")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing work: {e}")
    
    async def _add_prediction_to_queue(self, batch_id: str, predictions, api_key: Optional[str] = None, url: Optional[str] = None):
        """Add predictions to the processing queue with api_key for LSE client"""
        try:
            # Extract modelrun_id from current work message records if available
            modelrun_id = None
            if self.current_config and hasattr(self.current_config, 'records') and self.current_config.records:
                # Get modelrun_id from the first record (all records in a batch should have the same modelrun_id)
                first_record = self.current_config.records[0]
                modelrun_id = first_record.get('modelrun_id')
            
            prediction_data = {
                'batch_id': batch_id,
                'predictions': predictions,
                'api_key': api_key,  # Include api_key for LSE client creation
                'url': url,  # Include URL for LSE client creation
                'modelrun_id': modelrun_id,  # Include modelrun_id for LSE client creation
                'timestamp': datetime.now()
            }
            
            # Use put_nowait to avoid blocking the worker processor
            # If queue is full, this will raise QueueFull exception
            self.prediction_queue.put_nowait(prediction_data)
            
            logger.debug(f"Worker {self.worker_id}: Added prediction batch {batch_id} to queue (modelrun_id: {modelrun_id})")
            
        except asyncio.QueueFull:
            logger.warning(f"Worker {self.worker_id}: Queue full, dropping batch {batch_id}")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error adding prediction to queue: {e}")
    
    async def _should_take_work(self, work_message: WorkMessage) -> bool:
        """
        Determine if this worker should take this work
        Currently unused - can be used in future for more efficient work distribution
        """
        
        # If we have no current work, definitely take it
        if not self.current_config:
            return True
        
        # If we have the same config, take it (efficient)
        if self.current_config_hash == work_message.config_hash:
            return True
        
        # If we haven't switched configs recently, consider taking it
        if self._can_switch_config():
            # Simple strategy: take work if we've been idle for a while
            if self.last_processed_at:
                idle_time = (datetime.now() - self.last_processed_at).total_seconds()
                if idle_time > 30:  # Been idle for 30 seconds
                    return True
        
        return False
    
    def _can_switch_config(self) -> bool:
        """
        Check if we can switch configurations
        Currently unused - can be used in future for more efficient work distribution
        """
        if not self.last_config_switch:
            return True
        
        # Allow switching at most once every 60 seconds
        min_switch_interval = 60
        time_since_switch = (datetime.now() - self.last_config_switch).total_seconds()
        return time_since_switch > min_switch_interval
    
    async def _switch_configuration(self, work_message: WorkMessage):
        """
        Switch to new configuration for the worker
        This is done when the worker receives a new work message with different skills or runtime
        than the last message.
        """
        logger.info(f"Worker {self.worker_id}: Switching configuration")
        
        # Create new skills and runtime
        self.skills = self._create_skills_from_config(work_message.skills)
        self.runtime = self._create_runtime_from_config(work_message.runtime_params)
        
        self.config_switch_count += 1
        self.last_config_switch = datetime.now()
        
    
    def _create_skills_from_config(self, skills_config: List[Dict]):
        """Create skills from configuration"""
        from adala.skills import LinearSkillSet, TransformSkill, ClassificationSkill, EntityExtraction, LabelStudioSkill
        
        skill_instances = []
        for skill_config in skills_config:
            skill_type = skill_config.get('type')
            if skill_type == 'TransformSkill':
                skill = TransformSkill(**skill_config)
            elif skill_type == 'ClassificationSkill':
                skill = ClassificationSkill(**skill_config)
            elif skill_type == 'EntityExtraction':
                skill = EntityExtraction(**skill_config)
            elif skill_type == 'LabelStudioSkill':
                skill = LabelStudioSkill(**skill_config)
            else:
                logger.warning(f"Unknown skill type: {skill_type}, skipping")
                continue
            skill_instances.append(skill)
        
        return LinearSkillSet(skills=skill_instances)
    
    def _create_runtime_from_config(self, runtime_config: Dict):
        """Create runtime from configuration"""
        from adala.runtimes import AsyncLiteLLMChatRuntime, AsyncLiteLLMVisionRuntime, AsyncOpenAIChatRuntime, AsyncOpenAIVisionRuntime
        
        runtime_type = runtime_config.get('type', 'AsyncLiteLLMChatRuntime')
        
        if runtime_type == 'AsyncLiteLLMChatRuntime':
            return AsyncLiteLLMChatRuntime(**runtime_config)
        elif runtime_type == 'AsyncLiteLLMVisionRuntime':
            return AsyncLiteLLMVisionRuntime(**runtime_config)
        elif runtime_type == 'AsyncOpenAIChatRuntime':
            return AsyncOpenAIChatRuntime(**runtime_config)
        elif runtime_type == 'AsyncOpenAIVisionRuntime':
            return AsyncOpenAIVisionRuntime(**runtime_config)
        else:
            # Default to AsyncLiteLLMChatRuntime if type is not recognized
            logger.warning(f"Unknown runtime type: {runtime_type}, using AsyncLiteLLMChatRuntime")
            return AsyncLiteLLMChatRuntime(**runtime_config)
    
    async def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.environment:
            try:
                await self.environment.finalize()
                logger.info(f"Worker {self.worker_id}: Environment finalized")
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: Error finalizing environment: {e}")
            finally:
                self.environment = None
        
        # Clear references
        self.skills = None
        self.runtime = None
        self.current_config = None
        self.current_config_hash = None
        
        logger.info(f"Worker {self.worker_id}: Cleaned up")
    
    def get_status(self) -> Dict:
        """Get status of this worker"""
        return {
            'worker_id': self.worker_id,
            'is_running': self.is_running,
            'current_batch_id': self.current_config.batch_id if self.current_config else None,
            'current_config_hash': self.current_config_hash,
            'config_switch_count': self.config_switch_count,
            'last_processed_at': self.last_processed_at,
            'processed_batches': self.processed_batches,
            'last_config_switch': self.last_config_switch
        }
