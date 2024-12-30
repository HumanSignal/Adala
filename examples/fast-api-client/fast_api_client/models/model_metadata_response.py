from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.model_metadata_response_model_metadata import ModelMetadataResponseModelMetadata


T = TypeVar("T", bound="ModelMetadataResponse")


@_attrs_define
class ModelMetadataResponse:
    """
    Attributes:
        model_metadata (ModelMetadataResponseModelMetadata):
    """

    model_metadata: "ModelMetadataResponseModelMetadata"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        model_metadata = self.model_metadata.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "model_metadata": model_metadata,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.model_metadata_response_model_metadata import ModelMetadataResponseModelMetadata

        d = src_dict.copy()
        model_metadata = ModelMetadataResponseModelMetadata.from_dict(d.pop("model_metadata"))

        model_metadata_response = cls(
            model_metadata=model_metadata,
        )

        model_metadata_response.additional_properties = d
        return model_metadata_response

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
