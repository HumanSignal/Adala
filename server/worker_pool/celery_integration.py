# AI-MEMORY: Celery integration for per-worker processor architecture
# Benefits: 1) Automatic startup per worker 2) Shared work distribution 3) Simple scaling
# 4) Worker-specific status tracking 5) Clean separation via shared async queue

import asyncio
import logging
import os
import threading
from celery.signals import worker_process_init, worker_process_shutdown

from .worker_processor import WorkerProcessor
from .output_processor import OutputProcessor
from .restart_utils import WorkerRestartTrigger

logger = logging.getLogger(__name__)

class RestartWorkerException(Exception):
    """Custom exception to trigger worker restart from within processors"""
    def __init__(self, reason: str = "Unknown"):
        self.reason = reason
        super().__init__(f"Worker restart requested: {reason}")

class WorkerRestartManager:
    """Manages worker restart logic and statistics"""
    def __init__(self):
        self.restart_count = 0
        self.max_restarts = 10  # Prevent infinite restart loops
        
    def should_restart(self, exception: RestartWorkerException) -> bool:
        """Check if worker should be restarted based on restart count and reason"""
        if self.restart_count >= self.max_restarts:
            logger.error(f"Maximum restart limit ({self.max_restarts}) reached. Stopping worker.")
            return False
        
        logger.info(f"Restart #{self.restart_count + 1} requested: {exception.reason}")
        return True
        
    def increment_restart_count(self):
        """Increment restart counter"""
        self.restart_count += 1
        
    def reset_restart_count(self):
        """Reset restart counter (after successful run)"""
        self.restart_count = 0

# Global variables to track processors and their thread
worker_processor = None
output_processor = None
processor_thread = None
prediction_queue = None
restart_manager = None


@worker_process_init.connect
def worker_process_init_handler(**kwargs):
    """Initialize dedicated consumer worker and output processor with shared async queue"""
    global worker_processor, output_processor, processor_thread, prediction_queue, restart_manager

    def run_async_processors():
        """Run both processors in the same event loop with restart capability"""
        global restart_manager
        restart_manager = WorkerRestartManager()
        
        while True:  # Restart loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def start_processors():
                global worker_processor, output_processor, prediction_queue
                try:
                    # Create shared async queue for direct communication
                    prediction_queue = asyncio.Queue()  # Bounded queue for predictions
                    
                    # Create shared restart event for coordination
                    restart_event = asyncio.Event()

                    # Initialize both processors with shared queue and restart event
                    worker_processor = WorkerProcessor(prediction_queue=prediction_queue)
                    output_processor = OutputProcessor(prediction_queue=prediction_queue)
                    
                    # Add restart event to both processors
                    worker_processor.restart_event = restart_event
                    output_processor.restart_event = restart_event

                    # Create restart trigger and pass to processors
                    restart_trigger = WorkerRestartTrigger(RestartWorkerException, restart_event)
                    worker_processor.restart_trigger = restart_trigger
                    output_processor.restart_trigger = restart_trigger

                    # Initialize processors
                    await worker_processor.initialize()
                    await output_processor.initialize()  # No default result handler needed

                    logger.info("Worker pool processors initialized successfully")

                    # Run both processors concurrently
                    logger.info("Starting asyncio.gather for both processors")
                    results = await asyncio.gather(
                        worker_processor.run_forever(),
                        output_processor.run_forever(),
                        return_exceptions=True,
                    )
                    logger.info("asyncio.gather completed")
                    
                    # Check if any processor returned an exception
                    logger.info(f"Checking asyncio.gather results: {len(results)} results")
                    for i, result in enumerate(results):
                        logger.info(f"Result {i}: {type(result).__name__}: {result}")
                        if isinstance(result, RestartWorkerException):
                            logger.info(f"Restart requested from processor: {result.reason}")
                            raise result
                        elif isinstance(result, Exception):
                            logger.error(f"Processor error: {result}")
                            raise result

                except RestartWorkerException as e:
                    # Re-raise to be caught by outer restart loop
                    raise e
                except Exception as e:
                    logger.error(f"Processors error: {e}")
                    # Convert to restart exception for consistency
                    raise RestartWorkerException(f"Processor error: {e}")
                finally:
                    # Cleanup processors
                    if worker_processor:
                        worker_processor.is_running = False
                    if output_processor:
                        output_processor.is_running = False
                    
                    # Stop event loop
                    loop.stop()

            try:
                loop.run_until_complete(start_processors())
                # If we reach here without exception, break restart loop
                restart_manager.reset_restart_count()
                break
                
            except RestartWorkerException as e:
                # Handle restart request
                if restart_manager.should_restart(e):
                    restart_manager.increment_restart_count()
                    logger.info(f"Restarting worker due to: {e.reason}")
                    
                    # Cleanup current loop
                    try:
                        loop.close()
                    except:
                        pass
                    
                    # Brief delay before restart
                    import time
                    time.sleep(1)
                    continue  # Restart loop
                else:
                    logger.error("Maximum restarts reached. Stopping worker.")
                    break
                    
            except Exception as e:
                logger.error(f"Event loop error: {e}")
                # Try to restart on unexpected errors too
                restart_exception = RestartWorkerException(f"Event loop error: {e}")
                if restart_manager.should_restart(restart_exception):
                    restart_manager.increment_restart_count()
                    logger.info("Restarting worker due to event loop error")
                    
                    # Cleanup current loop
                    try:
                        loop.close()
                    except:
                        pass
                    
                    import time
                    time.sleep(1)
                    continue  # Restart loop
                else:
                    break
            finally:
                try:
                    loop.close()
                except:
                    pass

    # Start both processors in a separate thread
    processor_thread = threading.Thread(target=run_async_processors, daemon=True)
    processor_thread.start()

    logger.info(
        f"Worker and output processors started with restart capability for PID {os.getpid()}"
    )


@worker_process_shutdown.connect
def worker_process_shutdown_handler(**kwargs):
    """Cleanup processors when Celery worker shuts down"""
    global worker_processor, output_processor, processor_thread, prediction_queue, restart_manager

    # Signal both processors to stop
    if worker_processor:
        worker_processor.is_running = False
        logger.info(f"Worker processor shutdown for PID {os.getpid()}")

    if output_processor:
        output_processor.is_running = False
        logger.info(f"Output processor shutdown for PID {os.getpid()}")

    # Wait for the processor thread to finish (with timeout)
    if processor_thread and processor_thread.is_alive():
        processor_thread.join(timeout=5)
        if processor_thread.is_alive():
            logger.warning(
                f"Processor thread did not shut down gracefully for PID {os.getpid()}"
            )

    # Clear global references
    worker_processor = None
    output_processor = None
    processor_thread = None
    prediction_queue = None
    restart_manager = None

    logger.info(f"Worker pool processors cleaned up for PID {os.getpid()}")
