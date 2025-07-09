"""
Utility functions for triggering worker restarts from within processors.

This module provides a clean interface for processors to request worker restarts
when certain conditions are met (e.g., memory thresholds, error conditions).
"""

import logging

logger = logging.getLogger(__name__)


class WorkerRestartTrigger:
    """Helper class to trigger worker restarts from within processors"""

    def __init__(self):
        """
        Initialize the restart trigger.
        """
        pass

    def trigger_restart(self, reason: str):
        """
        Trigger a worker restart with the given reason.
        Forces immediate process termination for complete memory cleanup.

        Args:
            reason: Human-readable reason for the restart
        """
        logger.info(f"Triggering worker restart: {reason}")
        logger.info("Forcing immediate process termination for complete memory cleanup")
        logger.info("Celery will automatically restart the worker process")

        # Force immediate process termination - Celery will restart the worker process
        # This is the ONLY way to truly clean up native library memory leaks
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
