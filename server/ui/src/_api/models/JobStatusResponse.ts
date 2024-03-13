/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Status } from './Status';
/**
 *
 * Response model for getting the status of a job.
 *
 * Attributes:
 * status (str): The status of the job.
 * processed_total (List[int]): The total number of processed records and the total number of records in job.
 * Example: [10, 100] means 10% of the completeness.
 *
 */
export type JobStatusResponse = {
    status: Status;
};

