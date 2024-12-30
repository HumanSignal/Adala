from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent import Agent
    from ..models.improved_prompt_request_data_type_0_item import ImprovedPromptRequestDataType0Item


T = TypeVar("T", bound="ImprovedPromptRequest")


@_attrs_define
class ImprovedPromptRequest:
    """Request model for improving a prompt.

    Attributes:
        agent (Agent): Represents a customizable agent that can interact with environments,
            employ skills, and leverage memory and runtimes.

            Attributes:
                environment (Environment): The environment with which the agent interacts.
                skills (SkillSet): The skills possessed by the agent.
                memory (LongTermMemory, optional): The agent's long-term memory. Defaults to None.
                runtimes (Dict[str, Runtime], optional): The runtimes available to the agent. Defaults to predefined
            runtimes.
                default_runtime (str): The default runtime used by the agent. Defaults to 'openai'.
                teacher_runtimes (Dict[str, Runtime], optional): The runtimes available to the agent's teacher. Defaults to
            predefined runtimes.
                default_teacher_runtime (str): The default runtime used by the agent's teacher. Defaults to 'openai-gpt3'.

            Examples:
                >>> from adala.environments import StaticEnvironment
                >>> from adala.skills import LinearSkillSet, TransformSkill
                >>> from adala.agents import Agent
                >>> agent = Agent(skills=LinearSkillSet(skills=[TransformSkill()]), environment=StaticEnvironment())
                >>> agent.learn()  # starts the learning process
                >>> predictions = agent.run()  # runs the agent and returns the predictions
        skill_to_improve (str):
        input_variables (Union[None, Unset, list[str]]): List of variables available to use in the input template of the
            skill, in case any exist that are not currently used
        data (Union[None, Unset, list['ImprovedPromptRequestDataType0Item']]): Batch of data to run the skill on
        reapply (Union[Unset, bool]): Whether to reapply the skill to the data before improving the prompt Default:
            False.
        instructions (Union[None, Unset, str]): Instructions for the prompt improvement task Default: 'Improve current
            prompt'.
    """

    agent: "Agent"
    skill_to_improve: str
    input_variables: Union[None, Unset, list[str]] = UNSET
    data: Union[None, Unset, list["ImprovedPromptRequestDataType0Item"]] = UNSET
    reapply: Union[Unset, bool] = False
    instructions: Union[None, Unset, str] = "Improve current prompt"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent = self.agent.to_dict()

        skill_to_improve = self.skill_to_improve

        input_variables: Union[None, Unset, list[str]]
        if isinstance(self.input_variables, Unset):
            input_variables = UNSET
        elif isinstance(self.input_variables, list):
            input_variables = self.input_variables

        else:
            input_variables = self.input_variables

        data: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.data, Unset):
            data = UNSET
        elif isinstance(self.data, list):
            data = []
            for data_type_0_item_data in self.data:
                data_type_0_item = data_type_0_item_data.to_dict()
                data.append(data_type_0_item)

        else:
            data = self.data

        reapply = self.reapply

        instructions: Union[None, Unset, str]
        if isinstance(self.instructions, Unset):
            instructions = UNSET
        else:
            instructions = self.instructions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent": agent,
                "skill_to_improve": skill_to_improve,
            }
        )
        if input_variables is not UNSET:
            field_dict["input_variables"] = input_variables
        if data is not UNSET:
            field_dict["data"] = data
        if reapply is not UNSET:
            field_dict["reapply"] = reapply
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.agent import Agent
        from ..models.improved_prompt_request_data_type_0_item import ImprovedPromptRequestDataType0Item

        d = src_dict.copy()
        agent = Agent.from_dict(d.pop("agent"))

        skill_to_improve = d.pop("skill_to_improve")

        def _parse_input_variables(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                input_variables_type_0 = cast(list[str], data)

                return input_variables_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        input_variables = _parse_input_variables(d.pop("input_variables", UNSET))

        def _parse_data(data: object) -> Union[None, Unset, list["ImprovedPromptRequestDataType0Item"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                data_type_0 = []
                _data_type_0 = data
                for data_type_0_item_data in _data_type_0:
                    data_type_0_item = ImprovedPromptRequestDataType0Item.from_dict(data_type_0_item_data)

                    data_type_0.append(data_type_0_item)

                return data_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list["ImprovedPromptRequestDataType0Item"]], data)

        data = _parse_data(d.pop("data", UNSET))

        reapply = d.pop("reapply", UNSET)

        def _parse_instructions(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        instructions = _parse_instructions(d.pop("instructions", UNSET))

        improved_prompt_request = cls(
            agent=agent,
            skill_to_improve=skill_to_improve,
            input_variables=input_variables,
            data=data,
            reapply=reapply,
            instructions=instructions,
        )

        improved_prompt_request.additional_properties = d
        return improved_prompt_request

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
