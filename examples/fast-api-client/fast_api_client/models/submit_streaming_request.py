from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent import Agent
    from ..models.result_handler import ResultHandler


T = TypeVar("T", bound="SubmitStreamingRequest")


@_attrs_define
class SubmitStreamingRequest:
    """Request model for submitting a streaming job.

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
        result_handler (ResultHandler): Abstract base class for a result handler.
            This is a callable that is instantiated in `/submit-streaming` with any arguments that are needed, and then is
            called on each batch of results when it is finished being processed by the Agent (it consumes from the Kafka
            topic that the Agent produces to).

            It can be used as a connector to load results into a file or external service. If a ResultHandler is not used,
            the results will be discarded.

            Subclasses must implement the `__call__` method.

            The BaseModelInRegistry base class implements a factory pattern, allowing the "type" parameter to specify which
            subclass of ResultHandler to instantiate. For example:
            ```json
            result_handler: {
                "type": "DummyHandler",
                "other_model_field": "other_model_value",
                ...
            }
            ```
        task_name (Union[Unset, str]):  Default: 'streaming_parent_task'.
    """

    agent: "Agent"
    result_handler: "ResultHandler"
    task_name: Union[Unset, str] = "streaming_parent_task"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent = self.agent.to_dict()

        result_handler = self.result_handler.to_dict()

        task_name = self.task_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent": agent,
                "result_handler": result_handler,
            }
        )
        if task_name is not UNSET:
            field_dict["task_name"] = task_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.agent import Agent
        from ..models.result_handler import ResultHandler

        d = src_dict.copy()
        agent = Agent.from_dict(d.pop("agent"))

        result_handler = ResultHandler.from_dict(d.pop("result_handler"))

        task_name = d.pop("task_name", UNSET)

        submit_streaming_request = cls(
            agent=agent,
            result_handler=result_handler,
            task_name=task_name,
        )

        submit_streaming_request.additional_properties = d
        return submit_streaming_request

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
