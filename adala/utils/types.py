from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Union
from adala.utils.parse import parse_template
import asyncio
import time
import logging
import functools
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)


class BatchData(BaseModel):
    """
    Model for a batch of data submitted to a streaming job
    """

    job_id: str
    data: List[dict]


class ErrorResponseModel(BaseModel):
    message: str = Field(..., alias="_adala_message")
    details: str = Field(..., alias="_adala_details")

    model_config = ConfigDict(
        # omit other fields
        extra="ignore",
        # guard against name collisions with other fields
        populate_by_name=False,
    )


# Type variable for function return type
T = TypeVar("T")


def debug_time_it(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that measures the execution time of the wrapped function
    and logs it at debug level. Works with both synchronous and asynchronous functions.

    Args:
        func: The function to be wrapped (sync or async)

    Returns:
        The wrapped function with time measurement
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(
            "Function '%s' executed in %.4f seconds", func.__name__, execution_time
        )
        return result

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(
            "Async function '%s' executed in %.4f seconds",
            func.__name__,
            execution_time,
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return cast(Callable[..., T], async_wrapper)
    else:
        return cast(Callable[..., T], wrapper)
