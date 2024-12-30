from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.skill import Skill


T = TypeVar("T", bound="SkillSetSkillsType1")


@_attrs_define
class SkillSetSkillsType1:
    """ """

    additional_properties: dict[str, "Skill"] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.skill import Skill

        d = src_dict.copy()
        skill_set_skills_type_1 = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = Skill.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        skill_set_skills_type_1.additional_properties = additional_properties
        return skill_set_skills_type_1

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> "Skill":
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: "Skill") -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
