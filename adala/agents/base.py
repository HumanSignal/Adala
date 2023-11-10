from pydantic import BaseModel, Field, SkipValidation, field_validator, model_validator
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union, Tuple
from rich import print

from adala.environments.base import Environment, StaticEnvironment, GroundTruthSignal
from adala.datasets import Dataset, DataFrameDataset
from adala.runtimes.base import Runtime
from adala.runtimes._openai import OpenAIChatRuntime
from adala.runtimes import GuidanceRuntime, GuidanceModelType
from adala.skills._base import Skill
from adala.memories.base import Memory
from adala.skills.base import BaseSkill
from adala.skills.skillset import SkillSet, LinearSkillSet
from adala.utils.logs import print_dataframe, print_text, print_error
from adala.utils.internal_data import InternalDataFrame, InternalDataFrameConcat
from adala.skills.collection.analyze_errors import AnalyzeLLMPromptErrorsExactMatch
from adala.skills.collection.improve_llm import ImproveLLMInstructions


class Agent(BaseModel, ABC):
    """
    Represents a customizable agent that can interact with environments, 
    employ skills, and leverage memory and runtimes.

    Attributes:
        environment (Union[Dataset, Environment]): The environment with which the agent interacts.
        skills (Union[SkillSet, BaseSkill, List[BaseSkill], Dict[str, BaseSkill]]): The skills possessed by the agent.
        memory (LongTermMemory, optional): The agent's long-term memory. Defaults to None.
        runtimes (Dict[str, Runtime], optional): The runtimes available to the agent. Defaults to predefined runtimes.
        default_runtime (str): The default runtime used by the agent. Defaults to 'openai'.
        teacher_runtimes (Dict[str, Runtime], optional): The runtimes available to the agent's teacher. Defaults to predefined runtimes.
        default_teacher_runtime (str): The default runtime used by the agent's teacher. Defaults to 'openai-gpt3'.
    """
    
    environment: Union[InternalDataFrame, Dataset, Environment] = Field(default_factory=DataFrameDataset)
    skills: SkillSet

    memory: Memory = Field(default=None)
    runtimes: Optional[Dict[str, Runtime]] = Field(
        default_factory=lambda: {
            'openai': GuidanceRuntime()
            # 'openai': OpenAIChatRuntime(model='gpt-3.5-turbo'),
            # 'llama2': LLMRuntime(
            #     llm_runtime_type=LLMRuntimeModelType.Transformers,
            #     llm_params={
            #         'model': 'meta-llama/Llama-2-7b',
            #         'device': 'cuda:0',
            #     }
            # )
        }
    )
    teacher_runtimes: Optional[Dict[str, Runtime]] = Field(
        default_factory=lambda: {
            'openai-gpt3': OpenAIChatRuntime(model='gpt-3.5-turbo'),
            # 'openai-gpt4': OpenAIChatRuntime(model='gpt-4')
        }
    )
    default_runtime: str = 'openai'
    default_teacher_runtime: str = 'openai-gpt3'

    class Config:
        arbitrary_types_allowed = True

    def __rich__(self) -> str:
        """
        Returns a colorized and formatted representation of the Agent instance.

        Returns:
            str: A rich-formatted representation of the agent.
        """
        
        skill_names = ", ".join([skill.name for skill in self.skills.skills.values()])
        runtime_names = ", ".join(self.runtimes.keys())
        
        return (
            f"[bold blue]Agent Instance[/bold blue]\n\n"
            f"Environment: {self.environment.__class__.__name__}\n"
            f"Skills: {skill_names}\n"
            f"Runtimes: {runtime_names}\n"
            f"Default Runtime: {self.default_runtime}\n"
            f"Default Teacher Runtime: {self.default_teacher_runtime}"
        )

    @field_validator('environment')
    def environment_validator(cls, v) -> Environment:
        """
        Validates and possibly transforms the environment attribute.

        Args:
            v (Union[Dataset, Environment]): The environment value to validate.

        Returns:
            Environment: The validated environment.
        """
        if isinstance(v, InternalDataFrame):
            v = StaticEnvironment(df=v)
        return v

    @field_validator('skills', mode='before')
    def skills_validator(cls, v) -> SkillSet:
        """
        Validates and possibly transforms the skills attribute.

        Args:
            v (Union[SkillSet, BaseSkill, List[BaseSkill], Dict[str, BaseSkill]]): The skills value to validate.

        Returns:
            SkillSet: The validated set of skills.
        """
        if isinstance(v, SkillSet):
            return v
        elif isinstance(v, Skill):
            return LinearSkillSet(skills=[v])
        else:
            raise ValueError(f"skills must be of type SkillSet or Skill, not {type(v)}")

    @model_validator(mode='after')
    def verify_input_parameters(self):
        def _raise_default_runtime_error(val, runtime, runtimes, default_value):
            print_error(f"The Agent.{runtime} is set to {val}, "
                        f"but this runtime is not available in the list: {list(runtimes)}. "
                        f"Please choose one of the available runtimes and initialize the agent again, for example:\n\n"
                        f"agent = Agent(..., {runtime}='{default_value}')\n\n"
                        f"Make sure the default runtime is available in the list of runtimes. For example:\n\n"
                        f"agent = Agent(..., runtimes={{'{default_value}': OpenAIRuntime(model='gpt-4')}})\n\n")
            raise ValueError(f"default runtime {val} not found in provided runtimes.")

        if self.default_runtime not in self.runtimes:
            _raise_default_runtime_error(self.default_runtime, 'default_runtime', self.runtimes, 'openai')
        if self.default_teacher_runtime not in self.teacher_runtimes:
            _raise_default_runtime_error(self.default_teacher_runtime, 'default_teacher_runtime', self.teacher_runtimes, 'openai-gpt4')
        return self

    def get_runtime(self, runtime: Optional[str] = None) -> Runtime:
        """
        Retrieves the specified runtime or the default runtime if none is specified.

        Args:
            runtime (str, optional): The name of the runtime to retrieve. Defaults to None.

        Returns:
            Runtime: The requested runtime.

        Raises:
            ValueError: If the specified runtime is not found.
        """
        
        if runtime is None:
            runtime = self.default_runtime
        if runtime not in self.runtimes:
            raise ValueError(f'Runtime "{runtime}" not found.')
        return self.runtimes[runtime]

    def get_teacher_runtime(self, runtime: Optional[str] = None) -> Runtime:
        """
        Retrieves the specified teacher runtime or the default runtime if none is specified.

        Args:
            runtime (str, optional): The name of the runtime to retrieve. Defaults to None.

        Returns:
            Runtime: The requested runtime.

        Raises:
            ValueError: If the specified runtime is not found.
        """

        if runtime is None:
            runtime = self.default_teacher_runtime
        if runtime not in self.teacher_runtimes:
            raise ValueError(f'Teacher Runtime "{runtime}" not found.')
        return self.teacher_runtimes[runtime]

    def run(
        self,
        input: InternalDataFrame = None,
        runtime: Optional[str] = None
    ) -> InternalDataFrame:
        """
        Runs the agent on the specified dataset.

        Args:
            input (InternalDataFrame): The dataset to run the agent on.
            runtime (str, optional): The name of the runtime to use. Defaults to None, use the default runtime.

        Returns:
            InternalDataFrame: The dataset with the agent's predictions.
        """
        if input is None:
            input = self.environment.get_data_batch()
        runtime = self.get_runtime(runtime=runtime)
        predictions = self.skills.apply(input, runtime=runtime)
        return predictions

    def learn(
        self,
        learning_iterations: int = 3,
        accuracy_threshold: float = 0.9,
        update_memory: bool = True,
        request_environment_feedback: bool = True,
        wait_for_feedback: Optional[float] = True,
        num_feedbacks: Optional[int] = None,
        runtime: Optional[str] = None,
        teacher_runtime: Optional[str] = None,
    ) -> GroundTruthSignal:
        """
        Enables the agent to learn and improve its skills based on interactions with its environment.

        Args:
            learning_iterations (int, optional): The number of iterations for learning. Defaults to 3.
            accuracy_threshold (float, optional): The desired accuracy threshold to reach. Defaults to 0.9.
            update_memory (bool, optional): Flag to determine if memory should be updated after learning. Defaults to True.
            request_environment_feedback (bool, optional): Flag to determine if feedback should be requested from the environment. Defaults to True.
            wait_for_feedback (float, optional): The timeout in seconds to wait for environment feedback. Defaults to None.
            num_feedbacks (int, optional): The number of predictions to request feedback for. Defaults to None.
            runtime (str, optional): The runtime to be used for the learning process. Defaults to None.
            teacher_runtime (str, optional): The teacher runtime to be used for the learning process. Defaults to None.
        Returns:
            GroundTruthSignal: The ground truth signal.
        """
        
        runtime = self.get_runtime(runtime=runtime)
        teacher_runtime = self.get_teacher_runtime(runtime=teacher_runtime)

        data_batch = self.environment.get_data_batch()

        # Apply agent skills to dataset and get experience with predictions
        predictions = self.skills.apply(data_batch, runtime=runtime)

        ground_truth_signal = None

        for iteration in range(learning_iterations):
            print_text(f'\n\n=> Iteration #{iteration}: Comparing to ground truth, analyzing and improving ...')

            # Request feedback from environment is necessary
            if request_environment_feedback:
                self.environment.request_feedback(self.skills, predictions, num_feedbacks, wait_for_feedback)

            # Compare predictions to ground truth -> get ground truth signal
            ground_truth_signal = self.environment.compare_to_ground_truth(self.skills, predictions)

            print_text(f'Comparing predictions to ground truth data ...')
            print_dataframe(InternalDataFrameConcat([predictions, ground_truth_signal.match], axis=1))

            # Use ground truth signal to find the skill to improve
            accuracy = ground_truth_signal.get_accuracy()
            train_skill_name, train_skill_output = '', ''
            for skill_output, skill_name in self.skills.get_skill_outputs().items():
                if accuracy[skill_output] < accuracy_threshold:
                    train_skill_name, train_skill_output = skill_name, skill_output
                    break

            if not train_skill_name:
                print_text(f'No skill to improve found. Stopping learning process.')
                break

            train_skill = self.skills[train_skill_name]
            # select the worst performing skill
            print_text(f'Output to improve: "{train_skill_output}" (Skill="{train_skill_name}")\n'
                       f'Accuracy = {accuracy[train_skill_output] * 100:0.2f}%', style='bold red')

            skill_errors = ground_truth_signal.get_errors(train_skill_output).rename('_ground_truth')
            skill_errors = InternalDataFrameConcat((skill_errors, predictions), axis=1, join='inner')
            print(f'Errors for skill "{train_skill_name}":')
            print_dataframe(skill_errors)

            # 2. ANALYSIS PHASE: Analyze evaluation experience, optionally use long term memory

            teacher = LinearSkillSet(skills=[
                AnalyzeLLMPromptErrorsExactMatch(
                    name='analyze',
                    input_template=train_skill.input_template,
                    output_template='{error_report}',
                    initial_llm_instructions=train_skill.instructions,
                    prediction_column=train_skill_output,
                    ground_truth_column='_ground_truth',
                    field_schema=train_skill.field_schema,
                ),
                ImproveLLMInstructions(
                    name='improve',
                    old_instructions=train_skill.instructions,
                    field_schema=train_skill.field_schema,
                )
            ])

            result = teacher.apply(skill_errors, runtime=teacher_runtime)
            train_skill.instructions = result['new_instructions']
            print_text(f'Updated instructions for skill "{train_skill.name}":\n')
            print_text(train_skill.instructions, style='bold green')

            # 4. RE-APPLY PHASE: Re-apply skills to dataset
            print_text(f"Re-apply {train_skill.name} skill to dataset ...")
            self.skills[train_skill.name] = train_skill
            predictions = self.skills.apply(predictions, runtime=runtime, improved_skill=train_skill.name)

        print_text('Train is done!')
        return ground_truth_signal
