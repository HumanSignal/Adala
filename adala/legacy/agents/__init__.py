import warnings

warnings.warn(
    "The adala.agents module is deprecated and will be removed in a future version. "
    "Use adala.core.DataProcessor instead.",
    DeprecationWarning,
    stacklevel=2
)

from .base import Agent, create_agent_from_file, create_agent_from_dict
