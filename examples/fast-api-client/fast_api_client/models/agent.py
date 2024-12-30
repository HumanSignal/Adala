from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_runtimes import AgentRuntimes
    from ..models.agent_teacher_runtimes import AgentTeacherRuntimes
    from ..models.async_environment import AsyncEnvironment
    from ..models.environment import Environment
    from ..models.memory import Memory
    from ..models.skill_set import SkillSet


T = TypeVar("T", bound="Agent")


@_attrs_define
class Agent:
    """Represents a customizable agent that can interact with environments,
    employ skills, and leverage memory and runtimes.

    Attributes:
        environment (Environment): The environment with which the agent interacts.
        skills (SkillSet): The skills possessed by the agent.
        memory (LongTermMemory, optional): The agent's long-term memory. Defaults to None.
        runtimes (Dict[str, Runtime], optional): The runtimes available to the agent. Defaults to predefined runtimes.
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

        Attributes:
            skills (SkillSet): Represents a collection of interdependent skills aiming to achieve a specific goal.

                A skill set breaks down the path to achieve a goal into necessary precursor skills.
                Agents can evolve these skills either in parallel for tasks like self-consistency or
                sequentially for complex problem decompositions and causal reasoning. In the most generic
                cases, task decomposition can involve a graph-based approach.

                Attributes:
                    skills (Dict[str, Skill]): A dictionary of skills in the skill set.
            environment (Union['AsyncEnvironment', 'Environment', None, Unset]):
            memory (Union[Unset, Memory]): Base class for memories.
            runtimes (Union[Unset, AgentRuntimes]):
            default_runtime (Union[Unset, str]):  Default: 'default'.
            teacher_runtimes (Union[Unset, AgentTeacherRuntimes]):
            default_teacher_runtime (Union[Unset, str]):  Default: 'default'.
    """

    skills: "SkillSet"
    environment: Union["AsyncEnvironment", "Environment", None, Unset] = UNSET
    memory: Union[Unset, "Memory"] = UNSET
    runtimes: Union[Unset, "AgentRuntimes"] = UNSET
    default_runtime: Union[Unset, str] = "default"
    teacher_runtimes: Union[Unset, "AgentTeacherRuntimes"] = UNSET
    default_teacher_runtime: Union[Unset, str] = "default"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.async_environment import AsyncEnvironment
        from ..models.environment import Environment

        skills = self.skills.to_dict()

        environment: Union[None, Unset, dict[str, Any]]
        if isinstance(self.environment, Unset):
            environment = UNSET
        elif isinstance(self.environment, Environment):
            environment = self.environment.to_dict()
        elif isinstance(self.environment, AsyncEnvironment):
            environment = self.environment.to_dict()
        else:
            environment = self.environment

        memory: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.memory, Unset):
            memory = self.memory.to_dict()

        runtimes: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.runtimes, Unset):
            runtimes = self.runtimes.to_dict()

        default_runtime = self.default_runtime

        teacher_runtimes: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.teacher_runtimes, Unset):
            teacher_runtimes = self.teacher_runtimes.to_dict()

        default_teacher_runtime = self.default_teacher_runtime

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "skills": skills,
            }
        )
        if environment is not UNSET:
            field_dict["environment"] = environment
        if memory is not UNSET:
            field_dict["memory"] = memory
        if runtimes is not UNSET:
            field_dict["runtimes"] = runtimes
        if default_runtime is not UNSET:
            field_dict["default_runtime"] = default_runtime
        if teacher_runtimes is not UNSET:
            field_dict["teacher_runtimes"] = teacher_runtimes
        if default_teacher_runtime is not UNSET:
            field_dict["default_teacher_runtime"] = default_teacher_runtime

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.agent_runtimes import AgentRuntimes
        from ..models.agent_teacher_runtimes import AgentTeacherRuntimes
        from ..models.async_environment import AsyncEnvironment
        from ..models.environment import Environment
        from ..models.memory import Memory
        from ..models.skill_set import SkillSet

        d = src_dict.copy()
        skills = SkillSet.from_dict(d.pop("skills"))

        def _parse_environment(data: object) -> Union["AsyncEnvironment", "Environment", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                environment_type_0 = Environment.from_dict(data)

                return environment_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                environment_type_1 = AsyncEnvironment.from_dict(data)

                return environment_type_1
            except:  # noqa: E722
                pass
            return cast(Union["AsyncEnvironment", "Environment", None, Unset], data)

        environment = _parse_environment(d.pop("environment", UNSET))

        _memory = d.pop("memory", UNSET)
        memory: Union[Unset, Memory]
        if isinstance(_memory, Unset):
            memory = UNSET
        else:
            memory = Memory.from_dict(_memory)

        _runtimes = d.pop("runtimes", UNSET)
        runtimes: Union[Unset, AgentRuntimes]
        if isinstance(_runtimes, Unset):
            runtimes = UNSET
        else:
            runtimes = AgentRuntimes.from_dict(_runtimes)

        default_runtime = d.pop("default_runtime", UNSET)

        _teacher_runtimes = d.pop("teacher_runtimes", UNSET)
        teacher_runtimes: Union[Unset, AgentTeacherRuntimes]
        if isinstance(_teacher_runtimes, Unset):
            teacher_runtimes = UNSET
        else:
            teacher_runtimes = AgentTeacherRuntimes.from_dict(_teacher_runtimes)

        default_teacher_runtime = d.pop("default_teacher_runtime", UNSET)

        agent = cls(
            skills=skills,
            environment=environment,
            memory=memory,
            runtimes=runtimes,
            default_runtime=default_runtime,
            teacher_runtimes=teacher_runtimes,
            default_teacher_runtime=default_teacher_runtime,
        )

        agent.additional_properties = d
        return agent

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
