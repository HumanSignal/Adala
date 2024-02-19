/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 *
 * An abstract base class that defines the structure and required methods for an environment
 * in which machine learning models operate and are evaluated against ground truth data.
 *
 * Subclasses should implement methods to handle feedback requests, comparison to ground truth,
 * dataset conversion, and state persistence.
 *
 */
export type Environment = {
    type?: (string | null);
};

