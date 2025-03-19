"""
Adala Core - Simplified data processing with LLMs

This module provides a streamlined interface for batch processing tabular data 
through LLMs without unnecessary abstraction layers.
"""

from adala.core.processor import DataProcessor, Classifier
from adala.core.label_studio import LabelStudioProcessor
from adala.utils.internal_data import DataTable
from adala.runtimes.batch_llm import BatchLLMRuntime

__all__ = [
    'DataProcessor',
    'Classifier',
    'LabelStudioProcessor',
    'DataTable',
    'BatchLLMRuntime'
]