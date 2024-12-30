from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CostEstimate")


@_attrs_define
class CostEstimate:
    """
    Attributes:
        prompt_cost_usd (Union[None, Unset, float]):
        completion_cost_usd (Union[None, Unset, float]):
        total_cost_usd (Union[None, Unset, float]):
        is_error (Union[Unset, bool]):  Default: False.
        error_type (Union[None, Unset, str]):
        error_message (Union[None, Unset, str]):
    """

    prompt_cost_usd: Union[None, Unset, float] = UNSET
    completion_cost_usd: Union[None, Unset, float] = UNSET
    total_cost_usd: Union[None, Unset, float] = UNSET
    is_error: Union[Unset, bool] = False
    error_type: Union[None, Unset, str] = UNSET
    error_message: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        prompt_cost_usd: Union[None, Unset, float]
        if isinstance(self.prompt_cost_usd, Unset):
            prompt_cost_usd = UNSET
        else:
            prompt_cost_usd = self.prompt_cost_usd

        completion_cost_usd: Union[None, Unset, float]
        if isinstance(self.completion_cost_usd, Unset):
            completion_cost_usd = UNSET
        else:
            completion_cost_usd = self.completion_cost_usd

        total_cost_usd: Union[None, Unset, float]
        if isinstance(self.total_cost_usd, Unset):
            total_cost_usd = UNSET
        else:
            total_cost_usd = self.total_cost_usd

        is_error = self.is_error

        error_type: Union[None, Unset, str]
        if isinstance(self.error_type, Unset):
            error_type = UNSET
        else:
            error_type = self.error_type

        error_message: Union[None, Unset, str]
        if isinstance(self.error_message, Unset):
            error_message = UNSET
        else:
            error_message = self.error_message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if prompt_cost_usd is not UNSET:
            field_dict["prompt_cost_usd"] = prompt_cost_usd
        if completion_cost_usd is not UNSET:
            field_dict["completion_cost_usd"] = completion_cost_usd
        if total_cost_usd is not UNSET:
            field_dict["total_cost_usd"] = total_cost_usd
        if is_error is not UNSET:
            field_dict["is_error"] = is_error
        if error_type is not UNSET:
            field_dict["error_type"] = error_type
        if error_message is not UNSET:
            field_dict["error_message"] = error_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_prompt_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        prompt_cost_usd = _parse_prompt_cost_usd(d.pop("prompt_cost_usd", UNSET))

        def _parse_completion_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        completion_cost_usd = _parse_completion_cost_usd(d.pop("completion_cost_usd", UNSET))

        def _parse_total_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        total_cost_usd = _parse_total_cost_usd(d.pop("total_cost_usd", UNSET))

        is_error = d.pop("is_error", UNSET)

        def _parse_error_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        error_type = _parse_error_type(d.pop("error_type", UNSET))

        def _parse_error_message(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        error_message = _parse_error_message(d.pop("error_message", UNSET))

        cost_estimate = cls(
            prompt_cost_usd=prompt_cost_usd,
            completion_cost_usd=completion_cost_usd,
            total_cost_usd=total_cost_usd,
            is_error=is_error,
            error_type=error_type,
            error_message=error_message,
        )

        cost_estimate.additional_properties = d
        return cost_estimate

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
