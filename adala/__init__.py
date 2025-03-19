"""
Adala: Data Processing with LLMs

Adala provides tools for efficient data processing using Large Language Models.
"""

__version__ = "0.0.4dev"

# Import core components
from adala.core import DataProcessor, Classifier, LabelStudioProcessor, DataTable, BatchLLMRuntime

# Legacy imports (with deprecation warnings)
from adala.agents import Agent
from adala.environments import StaticEnvironment
from adala.skills import ClassificationSkill, TransformSkill, LabelStudioSkill

__all__ = [
    # Core components
    'DataProcessor',
    'Classifier',
    'LabelStudioProcessor',
    'DataTable',
    'BatchLLMRuntime',
    
    # Legacy components 
    'Agent',
    'StaticEnvironment',
    'ClassificationSkill',
    'TransformSkill',
    'LabelStudioSkill',
]