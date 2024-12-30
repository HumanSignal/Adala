from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.skill_set_skills_type_1 import SkillSetSkillsType1


T = TypeVar("T", bound="SkillSet")


@_attrs_define
class SkillSet:
    """Represents a collection of interdependent skills aiming to achieve a specific goal.

    A skill set breaks down the path to achieve a goal into necessary precursor skills.
    Agents can evolve these skills either in parallel for tasks like self-consistency or
    sequentially for complex problem decompositions and causal reasoning. In the most generic
    cases, task decomposition can involve a graph-based approach.

    Attributes:
        skills (Dict[str, Skill]): A dictionary of skills in the skill set.

        Attributes:
            skills (Union['SkillSetSkillsType1', list[Any]]):
    """

    skills: Union["SkillSetSkillsType1", list[Any]]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        skills: Union[dict[str, Any], list[Any]]
        if isinstance(self.skills, list):
            skills = self.skills

        else:
            skills = self.skills.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "skills": skills,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.skill_set_skills_type_1 import SkillSetSkillsType1

        d = src_dict.copy()

        def _parse_skills(data: object) -> Union["SkillSetSkillsType1", list[Any]]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                skills_type_0 = cast(list[Any], data)

                return skills_type_0
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            skills_type_1 = SkillSetSkillsType1.from_dict(data)

            return skills_type_1

        skills = _parse_skills(d.pop("skills"))

        skill_set = cls(
            skills=skills,
        )

        skill_set.additional_properties = d
        return skill_set

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
