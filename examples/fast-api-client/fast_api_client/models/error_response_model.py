from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ErrorResponseModel")


@_attrs_define
class ErrorResponseModel:
    """
    Attributes:
        field_adala_message (str):
        field_adala_details (str):
    """

    field_adala_message: str
    field_adala_details: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_adala_message = self.field_adala_message

        field_adala_details = self.field_adala_details

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "_adala_message": field_adala_message,
                "_adala_details": field_adala_details,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        field_adala_message = d.pop("_adala_message")

        field_adala_details = d.pop("_adala_details")

        error_response_model = cls(
            field_adala_message=field_adala_message,
            field_adala_details=field_adala_details,
        )

        error_response_model.additional_properties = d
        return error_response_model

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
