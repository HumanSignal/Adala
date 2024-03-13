/* generated using openapi-typescript-codegen -- do no edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 *
 * Abstract base class representing a skill.
 *
 * Provides methods to interact with and obtain information about skills.
 *
 * Attributes:
 * name (str): Unique name of the skill.
 * instructions (str): Instructs agent what to do with the input data.
 * input_template (str): Template for the input data.
 * output_template (str): Template for the output data.
 * description (Optional[str]): Description of the skill.
 * field_schema (Optional[Dict]): Field [JSON schema](https://json-schema.org/) to use in the templates. Defaults to all fields are strings,
 * i.e. analogous to {"field_n": {"type": "string"}}.
 * extra_fields (Optional[Dict[str, str]]): Extra fields to use in the templates. Defaults to None.
 * instructions_first (bool): Flag indicating if instructions should be executed before input. Defaults to True.
 * verbose (bool): Flag indicating if runtime outputs should be verbose. Defaults to False.
 * frozen (bool): Flag indicating if the skill is frozen. Defaults to False.
 * type (ClassVar[str]): Type of the skill.
 *
 */
export type Skill = {
    type?: (string | null);
    /**
     * Unique name of the skill
     */
    name: string;
    /**
     * Instructs agent what to do with the input data. Can use templating to refer to input fields.
     */
    instructions: string;
    /**
     * Template for the input data. Can use templating to refer to input parameters and perform data transformations.
     */
    input_template: string;
    /**
     * Template for the output data. Can use templating to refer to input parameters and perform data transformations
     */
    output_template: string;
    /**
     * Description of the skill. Can be used to retrieve skill from the library.
     */
    description?: (string | null);
    /**
     * JSON schema for the fields of the input and output data.
     */
    field_schema?: (Record<string, any> | null);
    /**
     * Flag indicating if instructions should be shown before the input data.
     */
    instructions_first?: boolean;
    /**
     * Flag indicating if the skill is frozen.
     */
    frozen?: boolean;
};

