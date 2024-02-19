from typing import Optional
from pydantic import BaseModel
from abc import ABC

_registry = {}


class BaseModelInRegistry(BaseModel, ABC):

    type: Optional[str] = None  # TODO: this is a workaround for the `type` being represented in OpenAPI schema. If you have a better idea, feel free to fix it

    def __init_subclass__(cls, **kwargs):
        global _registry

        super().__init_subclass__(**kwargs)
        type = cls.__name__

        if type in _registry:
            raise ValueError(f"Class type '{type}' is already registered. "
                             f"Available types: {list(_registry.keys())}")

        _registry[type] = cls

    @classmethod
    def create_from_registry(cls, type, **kwargs):

        if type not in _registry:
            raise ValueError(f"Class type '{type}' is not registered. "
                             f"Available types: {list(_registry.keys())}")
        return _registry[type](type=type, **kwargs)
