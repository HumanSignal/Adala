"""
Script to stop consumer connections from Celery workers.

This script is used to stop the consumer connections for specified Celery queues. If no queues are
specified, the script will stop consumer connections for all active queues. The script ensures that
all running jobs are completed before exiting.

Usage:
    python script_name.py [queue1 queue2 ...]

Arguments:
    queues (optional): Names of queues to stop consumer connections for. If not provided, all active
                       queues will be considered.

Functions:
    stop_consumer_for_queues(queues):
        Stops the consumer connections for the specified Celery queues. If no queues are specified,
        it inspects the Celery workers to find all active queues and stops the consumer connections
        for them. Waits until all running jobs are finished before exiting.

    main():
        Main function to parse command-line arguments and invoke the stop_consumer_for_queues function.
        Accepts a list of queue names as arguments. If no queue names are provided, it stops consumer
        connections for all active queues.

Entry Point:
    The script is intended to be run as a standalone program. The entry point calls the main function
    to parse arguments and stop consumer connections for queues.
"""

import time
import argparse
import logging
import os
import sys
from celery.app.control import Control
from stream_inference import app as celery

logger = logging.getLogger(__name__)


def stop_consumer_for_queues(queues):
    control = Control(celery)

    local_celery_host = f"celery@{os.environ['HOSTNAME']}"

    if not queues:
        inspect = control.inspect(destination=[local_celery_host])
        queues_info = inspect.active_queues()
        if queues_info:
            queues = {
                queue["name"]
                for worker_queues in queues_info.values()
                for queue in worker_queues
            }
        else:
            logger.info("No active queues found.")
            return

    for queue_name in queues:
        logger.info(f"Cancel consumer {local_celery_host} for queue: {queue_name}")
        control.cancel_consumer(queue_name, destination=[local_celery_host])

    inspect = control.inspect(destination=[local_celery_host])
    while True:
        active = inspect.active()
        running_jobs = [job for value in active.values() for job in value]
        if len(running_jobs) > 0:
            logger.info(
                f"{len(running_jobs)} jobs running: {', '.join(job['name'] for job in running_jobs)}"
            )
            time.sleep(10)
        else:
            logger.info("No running jobs")
            break

    sys.exit(0)  # Exit the script after all jobs have finished


def main():
    parser = argparse.ArgumentParser(
        description="Script to stop consumer connections from Celery worker"
    )
    parser.add_argument(
        "queues", nargs="*", help="Names of queues to stop consumer connections for"
    )
    args = parser.parse_args()

    stop_consumer_for_queues(args.queues)


if __name__ == "__main__":
    main()
