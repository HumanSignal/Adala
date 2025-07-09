"""
Utility functions for triggering worker restarts from within processors.

This module provides a clean interface for processors to request worker restarts
when certain conditions are met (e.g., memory thresholds, error conditions).
"""

import logging

logger = logging.getLogger(__name__)

class WorkerRestartTrigger:
    """Helper class to trigger worker restarts from within processors"""
    
    def __init__(self, restart_exception_class=None, restart_event=None):
        """
        Initialize the restart trigger.
        
        Args:
            restart_exception_class: The exception class to raise for restarts.
                                   Will be set by celery_integration.py
            restart_event: Shared asyncio.Event to coordinate restarts between processors
        """
        self.restart_exception_class = restart_exception_class
        self.restart_event = restart_event
        
    def trigger_restart(self, reason: str):
        """
        Trigger a worker restart with the given reason.
        Forces process termination for real memory cleanup.
        
        Args:
            reason: Human-readable reason for the restart
        """
        logger.info(f"Triggering worker restart: {reason}")
        
        # Set the restart event to signal all processors
        if self.restart_event:
            self.restart_event.set()
            logger.info("Restart event set - all processors will be notified")
        
        logger.info("Forcing process termination for real memory cleanup")
        logger.info("Celery will automatically restart the worker process")
        
        # Force process termination - Celery will restart the worker process
        # This is the ONLY way to truly clean up native library memory leaks
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

def trigger_restart_on_memory_threshold(current_memory_mb: float, threshold_mb: float = 500) -> bool:
    """
    Check if memory exceeds threshold and trigger restart if needed.
    
    Args:
        current_memory_mb: Current memory usage in MB
        threshold_mb: Memory threshold in MB (default 500MB)
        
    Returns:
        True if restart was triggered, False otherwise
    """
    if current_memory_mb > threshold_mb:
        # This will be caught by the processor and used to trigger restart
        return True
    return False

def trigger_restart_on_batch_count(current_count: int, max_batches: int = 100) -> bool:
    """
    Check if batch count exceeds limit and trigger restart if needed.
    
    Args:
        current_count: Current number of batches processed
        max_batches: Maximum batches before restart (default 100)
        
    Returns:
        True if restart was triggered, False otherwise
    """
    if current_count >= max_batches:
        return True
    return False 