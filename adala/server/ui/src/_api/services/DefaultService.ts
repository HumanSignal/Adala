/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { JobCancelRequest } from '../models/JobCancelRequest';
import type { JobStatusRequest } from '../models/JobStatusRequest';
import type { Response_JobCreated_ } from '../models/Response_JobCreated_';
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
    public submitSubmitPost({
        requestBody,
    }: {
        requestBody: SubmitRequest,
    }): CancelablePromise<Response_JobCreated_> {
        return this.httpRequest.request({
            method: 'POST',
            url: '/submit',
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
     * @returns any Successful Response
     * @throws ApiError
     */
    public getStatusGetStatusGet({
        requestBody,
    }: {
        requestBody: JobStatusRequest,
    }): CancelablePromise<any> {
        return this.httpRequest.request({
            method: 'GET',
            url: '/get-status',
            body: requestBody,
            mediaType: 'application/json',
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
     * @returns any Successful Response
     * @throws ApiError
     */
    public cancelJobCancelPost({
        requestBody,
    }: {
        requestBody: JobCancelRequest,
    }): CancelablePromise<any> {
        return this.httpRequest.request({
            method: 'POST',
            url: '/cancel',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
