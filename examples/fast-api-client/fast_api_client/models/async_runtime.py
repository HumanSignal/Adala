from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="AsyncRuntime")


@_attrs_define
class AsyncRuntime:
    """Async version of runtime that uses asyncio to process batch of records.

    Attributes:
        type_ (Union[None, Unset, str]):
        verbose (Union[Unset, bool]):  Default: False.
        batch_size (Union[None, Unset, int]):
        concurrent_clients (Union[None, Unset, int]):  Default: 1.
    """

    type_: Union[None, Unset, str] = UNSET
    verbose: Union[Unset, bool] = False
    batch_size: Union[None, Unset, int] = UNSET
    concurrent_clients: Union[None, Unset, int] = 1
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_: Union[None, Unset, str]
        if isinstance(self.type_, Unset):
            type_ = UNSET
        else:
            type_ = self.type_

        verbose = self.verbose

        batch_size: Union[None, Unset, int]
        if isinstance(self.batch_size, Unset):
            batch_size = UNSET
        else:
            batch_size = self.batch_size

        concurrent_clients: Union[None, Unset, int]
        if isinstance(self.concurrent_clients, Unset):
            concurrent_clients = UNSET
        else:
            concurrent_clients = self.concurrent_clients

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_
        if verbose is not UNSET:
            field_dict["verbose"] = verbose
        if batch_size is not UNSET:
            field_dict["batch_size"] = batch_size
        if concurrent_clients is not UNSET:
            field_dict["concurrent_clients"] = concurrent_clients

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

        verbose = d.pop("verbose", UNSET)

        def _parse_batch_size(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        batch_size = _parse_batch_size(d.pop("batch_size", UNSET))

        def _parse_concurrent_clients(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        concurrent_clients = _parse_concurrent_clients(d.pop("concurrent_clients", UNSET))

        async_runtime = cls(
            type_=type_,
            verbose=verbose,
            batch_size=batch_size,
            concurrent_clients=concurrent_clients,
        )

        async_runtime.additional_properties = d
        return async_runtime

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
