/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Skill } from './Skill';
/**
 *
 * Represents a collection of interdependent skills aiming to achieve a specific goal.
 *
 * A skill set breaks down the path to achieve a goal into necessary precursor skills.
 * Agents can evolve these skills either in parallel for tasks like self-consistency or
 * sequentially for complex problem decompositions and causal reasoning. In the most generic
 * cases, task decomposition can involve a graph-based approach.
 *
 * Attributes:
 * skills (Dict[str, Skill]): A dictionary of skills in the skill set.
 *
 */
export type SkillSet = {
    skills: Record<string, Skill>;
};

