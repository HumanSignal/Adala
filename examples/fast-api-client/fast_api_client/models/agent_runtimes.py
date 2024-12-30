from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.async_runtime import AsyncRuntime
    from ..models.runtime import Runtime


T = TypeVar("T", bound="AgentRuntimes")


@_attrs_define
class AgentRuntimes:
    """ """

    additional_properties: dict[str, Union["AsyncRuntime", "Runtime"]] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.runtime import Runtime

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            if isinstance(prop, Runtime):
                field_dict[prop_name] = prop.to_dict()
            else:
                field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.async_runtime import AsyncRuntime
        from ..models.runtime import Runtime

        d = src_dict.copy()
        agent_runtimes = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():

            def _parse_additional_property(data: object) -> Union["AsyncRuntime", "Runtime"]:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    additional_property_type_0 = Runtime.from_dict(data)

                    return additional_property_type_0
                except:  # noqa: E722
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                additional_property_type_1 = AsyncRuntime.from_dict(data)

                return additional_property_type_1

            additional_property = _parse_additional_property(prop_dict)

            additional_properties[prop_name] = additional_property

        agent_runtimes.additional_properties = additional_properties
        return agent_runtimes

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Union["AsyncRuntime", "Runtime"]:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Union["AsyncRuntime", "Runtime"]) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
