from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.agent import Agent
    from ..models.cost_estimate_request_substitutions_item import CostEstimateRequestSubstitutionsItem


T = TypeVar("T", bound="CostEstimateRequest")


@_attrs_define
class CostEstimateRequest:
    """
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
        prompt (str):
        substitutions (list['CostEstimateRequestSubstitutionsItem']):
    """

    agent: "Agent"
    prompt: str
    substitutions: list["CostEstimateRequestSubstitutionsItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent = self.agent.to_dict()

        prompt = self.prompt

        substitutions = []
        for substitutions_item_data in self.substitutions:
            substitutions_item = substitutions_item_data.to_dict()
            substitutions.append(substitutions_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent": agent,
                "prompt": prompt,
                "substitutions": substitutions,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.agent import Agent
        from ..models.cost_estimate_request_substitutions_item import CostEstimateRequestSubstitutionsItem

        d = src_dict.copy()
        agent = Agent.from_dict(d.pop("agent"))

        prompt = d.pop("prompt")

        substitutions = []
        _substitutions = d.pop("substitutions")
        for substitutions_item_data in _substitutions:
            substitutions_item = CostEstimateRequestSubstitutionsItem.from_dict(substitutions_item_data)

            substitutions.append(substitutions_item)

        cost_estimate_request = cls(
            agent=agent,
            prompt=prompt,
            substitutions=substitutions,
        )

        cost_estimate_request.additional_properties = d
        return cost_estimate_request

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
