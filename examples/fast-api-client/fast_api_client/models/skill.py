from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.skill_field_schema_type_0 import SkillFieldSchemaType0


T = TypeVar("T", bound="Skill")


@_attrs_define
class Skill:
    """Abstract base class representing a skill.

    Provides methods to interact with and obtain information about skills.

    Attributes:
        name (str): Unique name of the skill.
        instructions (str): Instructs agent what to do with the input data.
        input_template (str): Template for the input data.
        output_template (str): Template for the output data.
        description (Optional[str]): Description of the skill.
        field_schema (Optional[Dict]): Field [JSON schema](https://json-schema.org/) to use in the templates. Defaults
    to all fields are strings,
            i.e. analogous to {"field_n": {"type": "string"}}.
        extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
        instructions_first (bool): Flag indicating if instructions should be executed before input. Defaults to True.
        verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
        frozen (bool): Flag indicating if the skill is frozen. Defaults to False.
        response_model (Optional[Type[BaseModel]]): Pydantic-based response model for the skill. If used,
    `output_template` and `field_schema` are ignored. Note that using `response_model` will become the default in the
    future.
        type (ClassVar[str]): Type of the skill.

        Attributes:
            name (str): Unique name of the skill
            input_template (str): Template for the input data. Can use templating to refer to input parameters and perform
                data transformations.
            type_ (Union[None, Unset, str]):
            instructions (Union[Unset, str]): Instructs agent what to do with the input data. Can use templating to refer to
                input fields. Default: ''.
            output_template (Union[Unset, str]): Template for the output data. Can use templating to refer to input
                parameters and perform data transformations Default: ''.
            description (Union[None, Unset, str]): Description of the skill. Can be used to retrieve skill from the library.
                Default: ''.
            field_schema (Union['SkillFieldSchemaType0', None, Unset]): JSON schema for the fields of the input and output
                data.
            instructions_first (Union[Unset, bool]): Flag indicating if instructions should be shown before the input data.
                Default: True.
            frozen (Union[Unset, bool]): Flag indicating if the skill is frozen. Default: False.
            response_model (Union[Unset, Any]): Pydantic-based response model for the skill. If used, `output_template` and
                `field_schema` are ignored.
    """

    name: str
    input_template: str
    type_: Union[None, Unset, str] = UNSET
    instructions: Union[Unset, str] = ""
    output_template: Union[Unset, str] = ""
    description: Union[None, Unset, str] = ""
    field_schema: Union["SkillFieldSchemaType0", None, Unset] = UNSET
    instructions_first: Union[Unset, bool] = True
    frozen: Union[Unset, bool] = False
    response_model: Union[Unset, Any] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.skill_field_schema_type_0 import SkillFieldSchemaType0

        name = self.name

        input_template = self.input_template

        type_: Union[None, Unset, str]
        if isinstance(self.type_, Unset):
            type_ = UNSET
        else:
            type_ = self.type_

        instructions = self.instructions

        output_template = self.output_template

        description: Union[None, Unset, str]
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        field_schema: Union[None, Unset, dict[str, Any]]
        if isinstance(self.field_schema, Unset):
            field_schema = UNSET
        elif isinstance(self.field_schema, SkillFieldSchemaType0):
            field_schema = self.field_schema.to_dict()
        else:
            field_schema = self.field_schema

        instructions_first = self.instructions_first

        frozen = self.frozen

        response_model = self.response_model

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "input_template": input_template,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if output_template is not UNSET:
            field_dict["output_template"] = output_template
        if description is not UNSET:
            field_dict["description"] = description
        if field_schema is not UNSET:
            field_dict["field_schema"] = field_schema
        if instructions_first is not UNSET:
            field_dict["instructions_first"] = instructions_first
        if frozen is not UNSET:
            field_dict["frozen"] = frozen
        if response_model is not UNSET:
            field_dict["response_model"] = response_model

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.skill_field_schema_type_0 import SkillFieldSchemaType0

        d = src_dict.copy()
        name = d.pop("name")

        input_template = d.pop("input_template")

        def _parse_type_(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        type_ = _parse_type_(d.pop("type", UNSET))

        instructions = d.pop("instructions", UNSET)

        output_template = d.pop("output_template", UNSET)

        def _parse_description(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_field_schema(data: object) -> Union["SkillFieldSchemaType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                field_schema_type_0 = SkillFieldSchemaType0.from_dict(data)

                return field_schema_type_0
            except:  # noqa: E722
                pass
            return cast(Union["SkillFieldSchemaType0", None, Unset], data)

        field_schema = _parse_field_schema(d.pop("field_schema", UNSET))

        instructions_first = d.pop("instructions_first", UNSET)

        frozen = d.pop("frozen", UNSET)

        response_model = d.pop("response_model", UNSET)

        skill = cls(
            name=name,
            input_template=input_template,
            type_=type_,
            instructions=instructions,
            output_template=output_template,
            description=description,
            field_schema=field_schema,
            instructions_first=instructions_first,
            frozen=frozen,
            response_model=response_model,
        )

        skill.additional_properties = d
        return skill

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
