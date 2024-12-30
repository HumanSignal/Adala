from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_metadata_request_item_auth_info_type_0 import ModelMetadataRequestItemAuthInfoType0


T = TypeVar("T", bound="ModelMetadataRequestItem")


@_attrs_define
class ModelMetadataRequestItem:
    """
    Attributes:
        provider (str):
        model_name (str):
        auth_info (Union['ModelMetadataRequestItemAuthInfoType0', None, Unset]):
    """

    provider: str
    model_name: str
    auth_info: Union["ModelMetadataRequestItemAuthInfoType0", None, Unset] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.model_metadata_request_item_auth_info_type_0 import ModelMetadataRequestItemAuthInfoType0

        provider = self.provider

        model_name = self.model_name

        auth_info: Union[None, Unset, dict[str, Any]]
        if isinstance(self.auth_info, Unset):
            auth_info = UNSET
        elif isinstance(self.auth_info, ModelMetadataRequestItemAuthInfoType0):
            auth_info = self.auth_info.to_dict()
        else:
            auth_info = self.auth_info

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "model_name": model_name,
            }
        )
        if auth_info is not UNSET:
            field_dict["auth_info"] = auth_info

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.model_metadata_request_item_auth_info_type_0 import ModelMetadataRequestItemAuthInfoType0

        d = src_dict.copy()
        provider = d.pop("provider")

        model_name = d.pop("model_name")

        def _parse_auth_info(data: object) -> Union["ModelMetadataRequestItemAuthInfoType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                auth_info_type_0 = ModelMetadataRequestItemAuthInfoType0.from_dict(data)

                return auth_info_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ModelMetadataRequestItemAuthInfoType0", None, Unset], data)

        auth_info = _parse_auth_info(d.pop("auth_info", UNSET))

        model_metadata_request_item = cls(
            provider=provider,
            model_name=model_name,
            auth_info=auth_info,
        )

        model_metadata_request_item.additional_properties = d
        return model_metadata_request_item

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
