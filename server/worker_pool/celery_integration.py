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

logger = logging.getLogger(__name__)

# Global variables to track processors and their thread
worker_processor = None
output_processor = None
processor_thread = None
prediction_queue = None


@worker_process_init.connect
def worker_process_init_handler(**kwargs):
    """Initialize dedicated consumer worker and output processor with shared async queue"""
    global worker_processor, output_processor, processor_thread, prediction_queue

    def run_async_processors():
        """Run both processors in the same event loop with shared queue"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def start_processors():
            global worker_processor, output_processor, prediction_queue
            try:
                # Create shared async queue for direct communication
                prediction_queue = asyncio.Queue()  # Bounded queue for predictions

                # Initialize both processors with shared queue
                worker_processor = WorkerProcessor(prediction_queue=prediction_queue)
                output_processor = OutputProcessor(prediction_queue=prediction_queue)

                # Initialize processors
                await worker_processor.initialize()
                await output_processor.initialize()  # No default result handler needed

                logger.info("Worker pool processors initialized successfully")

                # Run both processors concurrently
                await asyncio.gather(
                    worker_processor.run_forever(),
                    output_processor.run_forever(),
                    return_exceptions=True,
                )

            except Exception as e:
                logger.error(f"Processors error: {e}")
            finally:
                loop.stop()

        try:
            loop.run_until_complete(start_processors())
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            loop.close()

    # Start both processors in a separate thread
    processor_thread = threading.Thread(target=run_async_processors, daemon=True)
    processor_thread.start()

    logger.info(
        f"Worker and output processors started with shared queue for PID {os.getpid()}"
    )


@worker_process_shutdown.connect
def worker_process_shutdown_handler(**kwargs):
    """Cleanup processors when Celery worker shuts down"""
    global worker_processor, output_processor, processor_thread, prediction_queue

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
    prediction_queue = None
