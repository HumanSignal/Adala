# AI-MEMORY: Status utilities for worker pool monitoring
# Benefits: 1) Easy monitoring of both processors 2) Debug information 3) Health checks
# 4) Performance metrics 5) Queue status monitoring

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_processor_status() -> Dict[str, Any]:
    """Get the status of both worker and output processors"""
    from .celery_integration import worker_processor, output_processor, prediction_queue
    
    status = {
        'worker_processor': None,
        'output_processor': None,
        'prediction_queue': None,
        'overall_status': 'not_initialized'
    }
    
    try:
        # Get worker processor status
        if worker_processor:
            status['worker_processor'] = worker_processor.get_status()
            logger.debug(f"Worker processor status: {status['worker_processor']}")
        else:
            logger.warning("Worker processor not initialized")
        
        # Get output processor status
        if output_processor:
            status['output_processor'] = output_processor.get_status()
            logger.debug(f"Output processor status: {status['output_processor']}")
        else:
            logger.warning("Output processor not initialized")
        
        # Get prediction queue status
        if prediction_queue:
            status['prediction_queue'] = {
                'size': prediction_queue.qsize(),
                'maxsize': prediction_queue.maxsize,
                'empty': prediction_queue.empty(),
                'full': prediction_queue.full()
            }
            logger.debug(f"Prediction queue status: {status['prediction_queue']}")
        else:
            logger.warning("Prediction queue not initialized")
        
        # Determine overall status
        worker_running = status['worker_processor'] and status['worker_processor'].get('is_running', False)
        output_running = status['output_processor'] and status['output_processor'].get('is_running', False)
        
        if worker_running and output_running:
            status['overall_status'] = 'running'
        elif worker_running or output_running:
            status['overall_status'] = 'partial'
        elif status['worker_processor'] or status['output_processor']:
            status['overall_status'] = 'stopped'
        else:
            status['overall_status'] = 'not_initialized'
            
    except Exception as e:
        logger.error(f"Error getting processor status: {e}")
        status['error'] = str(e)
    
    return status

def log_processor_status():
    """Log the current status of both processors with detailed information"""
    try:
        status = get_processor_status()
        
        logger.info("=== Worker Pool Status ===")
        logger.info(f"Overall Status: {status['overall_status']}")
        
        # Worker processor details
        if status['worker_processor']:
            wp = status['worker_processor']
            logger.info(f"Worker Processor: {wp['worker_id']} - Running: {wp['is_running']}")
            logger.info(f"  Processed Batches: {wp['processed_batches']}")
            logger.info(f"  Last Processed: {wp['last_processed_at']}")
            logger.info(f"  Current Batch: {wp['current_batch_id']}")
            logger.info(f"  Config Switches: {wp['config_switch_count']}")
        else:
            logger.warning("Worker Processor: Not initialized")
        
        # Output processor details
        if status['output_processor']:
            op = status['output_processor']
            logger.info(f"Output Processor: {op['processor_id']} - Running: {op['is_running']}")
            logger.info(f"  Processed Batches: {op['processed_batches']}")
            logger.info(f"  Last Processed: {op['last_processed_at']}")
            logger.info(f"  Queue Size: {op['queue_size']}/{op['queue_maxsize']}")
            
            # LSE client cache details
            if 'lse_client_cache' in op:
                cache_info = op['lse_client_cache']
                logger.info(f"  LSE Clients Cached: {cache_info['total_clients']}")
                if cache_info['client_ages']:
                    logger.info(f"  Client Ages: {cache_info['client_ages']}")
        else:
            logger.warning("Output Processor: Not initialized")
        
        # Prediction queue details
        if status['prediction_queue']:
            pq = status['prediction_queue']
            logger.info(f"Prediction Queue: Size {pq['size']}/{pq['maxsize']} - Empty: {pq['empty']}, Full: {pq['full']}")
        else:
            logger.warning("Prediction Queue: Not initialized")
        
        logger.info("=== End Status ===")
        
    except Exception as e:
        logger.error(f"Error logging processor status: {e}")

def is_healthy() -> bool:
    """Check if the processors are healthy (both running)"""
    try:
        status = get_processor_status()
        return status['overall_status'] == 'running'
    except Exception as e:
        logger.error(f"Error checking health: {e}")
        return False

def get_queue_metrics() -> Dict[str, Any]:
    """Get detailed queue metrics for monitoring"""
    from .celery_integration import prediction_queue
    
    metrics = {
        'queue_available': prediction_queue is not None,
        'queue_size': 0,
        'queue_maxsize': 0,
        'queue_empty': True,
        'queue_full': False
    }
    
    try:
        if prediction_queue:
            metrics.update({
                'queue_size': prediction_queue.qsize(),
                'queue_maxsize': prediction_queue.maxsize,
                'queue_empty': prediction_queue.empty(),
                'queue_full': prediction_queue.full()
            })
    except Exception as e:
        logger.error(f"Error getting queue metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics 