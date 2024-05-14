import time
import argparse
import logging
import sys
from celery.app.control import Control
from process_file import app as celery

logger = logging.getLogger(__name__)


def stop_consumer_for_queues(queues):
    control = Control(celery)

    if not queues:
        inspect = control.inspect()
        queues_info = inspect.active_queues()
        if queues_info:
            queues = {queue['name'] for worker_queues in queues_info.values() for queue in worker_queues}
        else:
            logger.info("No active queues found.")
            return

    for queue_name in queues:
        logger.info(f'Cancel consumer for queue: {queue_name}')
        control.cancel_consumer(queue_name)

    inspect = control.inspect()
    while True:
        active = inspect.active()
        running_jobs = [job for value in active.values() for job in value]
        if len(running_jobs) > 0:
            logger.info(f"{len(running_jobs)} jobs running: {', '.join(job['name'] for job in running_jobs)}")
            time.sleep(10)
        else:
            logger.info("No running jobs")
            break

    sys.exit(0)  # Exit the script after all jobs have finished


def main():
    parser = argparse.ArgumentParser(description='Script to stop consumer connections from Celery worker')
    parser.add_argument('queues', nargs='*', help='Names of queues to stop consumer connections for')
    args = parser.parse_args()

    stop_consumer_for_queues(args.queues)


if __name__ == "__main__":
    main()
