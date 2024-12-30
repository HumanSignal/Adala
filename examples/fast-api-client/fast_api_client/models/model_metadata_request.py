from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.model_metadata_request_item import ModelMetadataRequestItem


T = TypeVar("T", bound="ModelMetadataRequest")


@_attrs_define
class ModelMetadataRequest:
    """
    Attributes:
        models (list['ModelMetadataRequestItem']):
    """

    models: list["ModelMetadataRequestItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        models = []
        for models_item_data in self.models:
            models_item = models_item_data.to_dict()
            models.append(models_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "models": models,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.model_metadata_request_item import ModelMetadataRequestItem

        d = src_dict.copy()
        models = []
        _models = d.pop("models")
        for models_item_data in _models:
            models_item = ModelMetadataRequestItem.from_dict(models_item_data)

            models.append(models_item)

        model_metadata_request = cls(
            models=models,
        )

        model_metadata_request.additional_properties = d
        return model_metadata_request

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
