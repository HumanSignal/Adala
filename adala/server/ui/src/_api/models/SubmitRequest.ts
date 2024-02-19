/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Agent } from './Agent';
/**
 *
 * Request model for submitting a job.
 *
 * Attributes:
 * agent (Agent): The agent to be used for the task. Example of serialized agent:
 * {
     * "skills": [{
         * "type": "classification",
         * "name": "text_classifier",
         * "instructions": "Classify the text.",
         * "input_template": "Text: {text}",
         * "output_template": "Classification result: {label}",
         * "labels": {
             * "label": ['label1', 'label2', 'label3']
             * }
             * }],
             * "runtimes": {
                 * "default": {
                     * "type": "openai-chat",
                     * "model": "gpt-3.5-turbo",
                     * "api_key": "..."
                     * }
                     * }
                     * }
                     * task_name (str): The name of the task to be executed by the agent.
                     *
                     */
                    export type SubmitRequest = {
                        agent: Agent;
                        task_name?: string;
                    };

