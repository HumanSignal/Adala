import logging
from typing import Optional
from pydantic import BaseModel, field_serializer
from abc import ABC

logger = logging.getLogger(__name__)

_registry = {}


class BaseModelInRegistry(BaseModel, ABC):
    type: Optional[
        str
    ] = None  # TODO: this is a workaround for the `type` being represented in OpenAPI schema. If you have a better idea, feel free to fix it

    @field_serializer("type")
    def serialize_type(self, v: str) -> str:
        if v is None:
            v = self.__class__.__name__
        return v

    def __init_subclass__(cls, **kwargs):
        global _registry

        super().__init_subclass__(**kwargs)
        type = cls.__name__

        if type in _registry:
            logger.warning(
                f"Class type '{type}' is already registered. "
                f"Available types: {list(_registry.keys())}"
            )

        _registry[type] = cls

    @classmethod
    def create_from_registry(cls, type, **kwargs):
        if type not in _registry:
            raise ValueError(
                f"Class type '{type}' is not registered. "
                f"Available types: {list(_registry.keys())}"
            )
        obj = _registry[type](type=type, **kwargs)
        return obj
