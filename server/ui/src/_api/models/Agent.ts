/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AsyncEnvironment } from './AsyncEnvironment';
import type { AsyncRuntime } from './AsyncRuntime';
import type { Environment } from './Environment';
import type { Memory } from './Memory';
import type { Runtime } from './Runtime';
import type { Skill } from './Skill';
import type { SkillSet } from './SkillSet';
/**
 *
 * Represents a customizable agent that can interact with environments,
 * employ skills, and leverage memory and runtimes.
 *
 * Attributes:
 * environment (Environment): The environment with which the agent interacts.
 * skills (Union[SkillSet, List[Skill]]): The skills possessed by the agent.
 * memory (LongTermMemory, optional): The agent's long-term memory. Defaults to None.
 * runtimes (Dict[str, Runtime], optional): The runtimes available to the agent. Defaults to predefined runtimes.
 * default_runtime (str): The default runtime used by the agent. Defaults to 'openai'.
 * teacher_runtimes (Dict[str, Runtime], optional): The runtimes available to the agent's teacher. Defaults to predefined runtimes.
 * default_teacher_runtime (str): The default runtime used by the agent's teacher. Defaults to 'openai-gpt3'.
 *
 * Examples:
 * >>> from adala.environments import StaticEnvironment
 * >>> from adala.skills import LinearSkillSet, TransformSkill
 * >>> from adala.agents import Agent
 * >>> agent = Agent(skills=LinearSkillSet(skills=[TransformSkill()]), environment=StaticEnvironment())
 * >>> agent.learn()  # starts the learning process
 * >>> predictions = agent.run()  # runs the agent and returns the predictions
 *
 */
export type Agent = {
    environment?: (Environment | AsyncEnvironment | null);
    skills: (Skill | SkillSet);
    memory?: Memory;
    runtimes?: Record<string, (Runtime | AsyncRuntime)>;
    teacher_runtimes?: Record<string, Runtime>;
    default_runtime?: string;
    default_teacher_runtime?: string;
};

