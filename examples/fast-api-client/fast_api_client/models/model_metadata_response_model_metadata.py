from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.model_metadata_response_model_metadata_additional_property import (
        ModelMetadataResponseModelMetadataAdditionalProperty,
    )


T = TypeVar("T", bound="ModelMetadataResponseModelMetadata")


@_attrs_define
class ModelMetadataResponseModelMetadata:
    """ """

    additional_properties: dict[str, "ModelMetadataResponseModelMetadataAdditionalProperty"] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.model_metadata_response_model_metadata_additional_property import (
            ModelMetadataResponseModelMetadataAdditionalProperty,
        )

        d = src_dict.copy()
        model_metadata_response_model_metadata = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = ModelMetadataResponseModelMetadataAdditionalProperty.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        model_metadata_response_model_metadata.additional_properties = additional_properties
        return model_metadata_response_model_metadata

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "ModelMetadataResponseModelMetadataAdditionalProperty":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "ModelMetadataResponseModelMetadataAdditionalProperty") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
