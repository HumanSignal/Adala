from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ResultHandler")


@_attrs_define
class ResultHandler:
    """Abstract base class for a result handler.
    This is a callable that is instantiated in `/submit-streaming` with any arguments that are needed, and then is
    called on each batch of results when it is finished being processed by the Agent (it consumes from the Kafka topic
    that the Agent produces to).

    It can be used as a connector to load results into a file or external service. If a ResultHandler is not used, the
    results will be discarded.

    Subclasses must implement the `__call__` method.

    The BaseModelInRegistry base class implements a factory pattern, allowing the "type" parameter to specify which
    subclass of ResultHandler to instantiate. For example:
    ```json
    result_handler: {
        "type": "DummyHandler",
        "other_model_field": "other_model_value",
        ...
    }
    ```

        Attributes:
            type_ (Union[None, Unset, str]):
    """

    type_: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_: Union[None, Unset, str]
        if isinstance(self.type_, Unset):
            type_ = UNSET
        else:
            type_ = self.type_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_type_(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        type_ = _parse_type_(d.pop("type", UNSET))

        result_handler = cls(
            type_=type_,
        )

        result_handler.additional_properties = d
        return result_handler

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
