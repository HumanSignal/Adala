from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="PromptImprovementSkillResponseModel")


@_attrs_define
class PromptImprovementSkillResponseModel:
    """
    Attributes:
        reasoning (str): The reasoning for the changes made to the prompt
        new_prompt_title (str): The new short title for the prompt
        new_prompt_content (str):
    """

    reasoning: str
    new_prompt_title: str
    new_prompt_content: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reasoning = self.reasoning

        new_prompt_title = self.new_prompt_title

        new_prompt_content = self.new_prompt_content

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "reasoning": reasoning,
                "new_prompt_title": new_prompt_title,
                "new_prompt_content": new_prompt_content,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        reasoning = d.pop("reasoning")

        new_prompt_title = d.pop("new_prompt_title")

        new_prompt_content = d.pop("new_prompt_content")

        prompt_improvement_skill_response_model = cls(
            reasoning=reasoning,
            new_prompt_title=new_prompt_title,
            new_prompt_content=new_prompt_content,
        )

        prompt_improvement_skill_response_model.additional_properties = d
        return prompt_improvement_skill_response_model

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
