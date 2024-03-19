/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Response_JobCancelResponse_ } from '../models/Response_JobCancelResponse_';
import type { Response_JobCreated_ } from '../models/Response_JobCreated_';
import type { Response_JobStatusResponse_ } from '../models/Response_JobStatusResponse_';
import type { SubmitRequest } from '../models/SubmitRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import type { BaseHttpRequest } from '../core/BaseHttpRequest';
export class DefaultService {
    constructor(public readonly httpRequest: BaseHttpRequest) {}
    /**
     * Get Index
     * @returns any Successful Response
     * @throws ApiError
     */
    public getIndexGet(): CancelablePromise<any> {
        return this.httpRequest.request({
            method: 'GET',
            url: '/',
        });
    }
    /**
     * Submit
     * Submit a request to execute task `request.task_name` in celery.
     *
     * Args:
     * request (SubmitRequest): The request model for submitting a job.
     *
     * Returns:
     * Response[JobCreated]: The response model for a job created.
     * @returns Response_JobCreated_ Successful Response
     * @throws ApiError
     */
    public submitJobsSubmitPost({
        requestBody,
    }: {
        requestBody: SubmitRequest,
    }): CancelablePromise<Response_JobCreated_> {
        return this.httpRequest.request({
            method: 'POST',
            url: '/jobs/submit',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Status
     * Get the status of a job.
     *
     * Args:
     * request (JobStatusRequest): The request model for getting the status of a job.
     *
     * Returns:
     * JobStatusResponse: The response model for getting the status of a job.
     * @returns Response_JobStatusResponse_ Successful Response
     * @throws ApiError
     */
    public getStatusJobsJobIdGet({
        jobId,
    }: {
        jobId: any,
    }): CancelablePromise<Response_JobStatusResponse_> {
        return this.httpRequest.request({
            method: 'GET',
            url: '/jobs/{job_id}',
            path: {
                'job_id': jobId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Cancel Job
     * Cancel a job.
     *
     * Args:
     * request (JobCancelRequest): The request model for cancelling a job.
     *
     * Returns:
     * JobCancelResponse: The response model for cancelling a job.
     * @returns Response_JobCancelResponse_ Successful Response
     * @throws ApiError
     */
    public cancelJobJobsJobIdDelete({
        jobId,
    }: {
        jobId: any,
    }): CancelablePromise<Response_JobCancelResponse_> {
        return this.httpRequest.request({
            method: 'DELETE',
            url: '/jobs/{job_id}',
            path: {
                'job_id': jobId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
