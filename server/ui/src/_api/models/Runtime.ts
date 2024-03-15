/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 *
 * Base class representing a generic runtime environment.
 *
 * Attributes:
 * verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
 * batch_size (Optional[int]): The batch size to use for processing records. Defaults to None.
 *
 */
export type Runtime = {
    type?: (string | null);
    verbose?: boolean;
    batch_size?: (number | null);
};

