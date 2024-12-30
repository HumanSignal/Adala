from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.error_response_model import ErrorResponseModel
    from ..models.prompt_improvement_skill_response_model import PromptImprovementSkillResponseModel


T = TypeVar("T", bound="ImprovedPromptResponse")


@_attrs_define
class ImprovedPromptResponse:
    """
    Attributes:
        output (Union['ErrorResponseModel', 'PromptImprovementSkillResponseModel']):
        field_prompt_tokens (Union[Unset, int]):
        field_completion_tokens (Union[Unset, int]):
        field_prompt_cost_usd (Union[None, Unset, float]):
        field_completion_cost_usd (Union[None, Unset, float]):
        field_total_cost_usd (Union[None, Unset, float]):
    """

    output: Union["ErrorResponseModel", "PromptImprovementSkillResponseModel"]
    field_prompt_tokens: Union[Unset, int] = UNSET
    field_completion_tokens: Union[Unset, int] = UNSET
    field_prompt_cost_usd: Union[None, Unset, float] = UNSET
    field_completion_cost_usd: Union[None, Unset, float] = UNSET
    field_total_cost_usd: Union[None, Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.prompt_improvement_skill_response_model import PromptImprovementSkillResponseModel

        output: dict[str, Any]
        if isinstance(self.output, PromptImprovementSkillResponseModel):
            output = self.output.to_dict()
        else:
            output = self.output.to_dict()

        field_prompt_tokens = self.field_prompt_tokens

        field_completion_tokens = self.field_completion_tokens

        field_prompt_cost_usd: Union[None, Unset, float]
        if isinstance(self.field_prompt_cost_usd, Unset):
            field_prompt_cost_usd = UNSET
        else:
            field_prompt_cost_usd = self.field_prompt_cost_usd

        field_completion_cost_usd: Union[None, Unset, float]
        if isinstance(self.field_completion_cost_usd, Unset):
            field_completion_cost_usd = UNSET
        else:
            field_completion_cost_usd = self.field_completion_cost_usd

        field_total_cost_usd: Union[None, Unset, float]
        if isinstance(self.field_total_cost_usd, Unset):
            field_total_cost_usd = UNSET
        else:
            field_total_cost_usd = self.field_total_cost_usd

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "output": output,
            }
        )
        if field_prompt_tokens is not UNSET:
            field_dict["_prompt_tokens"] = field_prompt_tokens
        if field_completion_tokens is not UNSET:
            field_dict["_completion_tokens"] = field_completion_tokens
        if field_prompt_cost_usd is not UNSET:
            field_dict["_prompt_cost_usd"] = field_prompt_cost_usd
        if field_completion_cost_usd is not UNSET:
            field_dict["_completion_cost_usd"] = field_completion_cost_usd
        if field_total_cost_usd is not UNSET:
            field_dict["_total_cost_usd"] = field_total_cost_usd

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.error_response_model import ErrorResponseModel
        from ..models.prompt_improvement_skill_response_model import PromptImprovementSkillResponseModel

        d = src_dict.copy()

        def _parse_output(data: object) -> Union["ErrorResponseModel", "PromptImprovementSkillResponseModel"]:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_type_0 = PromptImprovementSkillResponseModel.from_dict(data)

                return output_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            output_type_1 = ErrorResponseModel.from_dict(data)

            return output_type_1

        output = _parse_output(d.pop("output"))

        field_prompt_tokens = d.pop("_prompt_tokens", UNSET)

        field_completion_tokens = d.pop("_completion_tokens", UNSET)

        def _parse_field_prompt_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        field_prompt_cost_usd = _parse_field_prompt_cost_usd(d.pop("_prompt_cost_usd", UNSET))

        def _parse_field_completion_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        field_completion_cost_usd = _parse_field_completion_cost_usd(d.pop("_completion_cost_usd", UNSET))

        def _parse_field_total_cost_usd(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        field_total_cost_usd = _parse_field_total_cost_usd(d.pop("_total_cost_usd", UNSET))

        improved_prompt_response = cls(
            output=output,
            field_prompt_tokens=field_prompt_tokens,
            field_completion_tokens=field_completion_tokens,
            field_prompt_cost_usd=field_prompt_cost_usd,
            field_completion_cost_usd=field_completion_cost_usd,
            field_total_cost_usd=field_total_cost_usd,
        )

        improved_prompt_response.additional_properties = d
        return improved_prompt_response

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
