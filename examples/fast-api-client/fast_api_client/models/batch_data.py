from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.batch_data_data_item import BatchDataDataItem


T = TypeVar("T", bound="BatchData")


@_attrs_define
class BatchData:
    """Model for a batch of data submitted to a streaming job

    Attributes:
        job_id (str):
        data (list['BatchDataDataItem']):
    """

    job_id: str
    data: list["BatchDataDataItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        job_id = self.job_id

        data = []
        for data_item_data in self.data:
            data_item = data_item_data.to_dict()
            data.append(data_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "job_id": job_id,
                "data": data,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.batch_data_data_item import BatchDataDataItem

        d = src_dict.copy()
        job_id = d.pop("job_id")

        data = []
        _data = d.pop("data")
        for data_item_data in _data:
            data_item = BatchDataDataItem.from_dict(data_item_data)

            data.append(data_item)

        batch_data = cls(
            job_id=job_id,
            data=data,
        )

        batch_data.additional_properties = d
        return batch_data

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
