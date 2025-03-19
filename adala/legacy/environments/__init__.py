import warnings

warnings.warn(
    "The adala.environments module is deprecated and will be removed in a future version. "
    "Use pandas.DataFrame or adala.core.DataTable directly instead.",
    DeprecationWarning,
    stacklevel=2
)

from .base import Environment, AsyncEnvironment, EnvironmentFeedback
from .static_env import StaticEnvironment
from .console import ConsoleEnvironment
from .web import WebStaticEnvironment
from .code_env import SimpleCodeValidationEnvironment
from .kafka import AsyncKafkaEnvironment
