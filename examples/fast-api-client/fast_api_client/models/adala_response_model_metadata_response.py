from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_metadata_response import ModelMetadataResponse


T = TypeVar("T", bound="AdalaResponseModelMetadataResponse")


@_attrs_define
class AdalaResponseModelMetadataResponse:
    """
    Attributes:
        data (ModelMetadataResponse):
        success (Union[Unset, bool]):  Default: True.
        message (Union[None, Unset, str]):
        errors (Union[None, Unset, list[Any]]):
    """

    data: "ModelMetadataResponse"
    success: Union[Unset, bool] = True
    message: Union[None, Unset, str] = UNSET
    errors: Union[None, Unset, list[Any]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = self.data.to_dict()

        success = self.success

        message: Union[None, Unset, str]
        if isinstance(self.message, Unset):
            message = UNSET
        else:
            message = self.message

        errors: Union[None, Unset, list[Any]]
        if isinstance(self.errors, Unset):
            errors = UNSET
        elif isinstance(self.errors, list):
            errors = self.errors

        else:
            errors = self.errors

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "data": data,
            }
        )
        if success is not UNSET:
            field_dict["success"] = success
        if message is not UNSET:
            field_dict["message"] = message
        if errors is not UNSET:
            field_dict["errors"] = errors

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.model_metadata_response import ModelMetadataResponse

        d = src_dict.copy()
        data = ModelMetadataResponse.from_dict(d.pop("data"))

        success = d.pop("success", UNSET)

        def _parse_message(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        message = _parse_message(d.pop("message", UNSET))

        def _parse_errors(data: object) -> Union[None, Unset, list[Any]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                errors_type_0 = cast(list[Any], data)

                return errors_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[Any]], data)

        errors = _parse_errors(d.pop("errors", UNSET))

        adala_response_model_metadata_response = cls(
            data=data,
            success=success,
            message=message,
            errors=errors,
        )

        adala_response_model_metadata_response.additional_properties = d
        return adala_response_model_metadata_response

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
