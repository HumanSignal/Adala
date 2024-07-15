import os
import difflib
import asyncio
from rich import print

from typing import Optional, Dict, Any, List
from pydantic import field_validator
from openai import OpenAI, NotFoundError, AsyncOpenAI
from pydantic import Field, computed_field, ConfigDict
from .base import Runtime, AsyncRuntime
from adala.utils.logs import print_error
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from adala.utils.parse import parse_template, partial_str_format
from adala.utils.matching import match_options
from tenacity import retry, stop_after_attempt, wait_random_exponential
import httpx


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def async_create_completion(
    model: str,
    user_prompt: str,
    client: AsyncOpenAI,
    system_prompt: str = None,
    openai_api_key: str = None,
    instruction_first: bool = True,
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Async version of create_completion function with error handling and session timeout.

    Args:
        model: OpenAI model name.
        user_prompt: User prompt.
        client: Async OpenAI client.
        system_prompt: System prompt.
        openai_api_key: OpenAI API key (if not set, will use OPENAI_API_KEY environment variable).
        instruction_first: Whether to put instructions first.
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.

    Returns:
        Dict[str, Any]: OpenAI response or error message.
    """
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        if instruction_first:
            messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages[0]["content"] += system_prompt

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        completion_text = completion.choices[0].message.content
        return {
            "text": completion_text,
            "_adala_error": False,
            "_adala_message": None,
            "_adala_details": None,
        }
    except Exception as e:
        # Handle other exceptions
        return {
            "text": None,
            "_adala_error": True,
            "_adala_message": type(e).__name__,
            "_adala_details": str(e),
        }


async def async_concurrent_create_completion(
    prompts,
    client,
    instruction_first,
    openai_model,
    max_tokens,
    temperature,
):
    tasks = [
        asyncio.ensure_future(
            async_create_completion(
                client=client,
                user_prompt=prompt["user"],
                system_prompt=prompt["system"],
                model=openai_model,
                max_tokens=max_tokens,
                temperature=temperature,
                instruction_first=instruction_first,
            )
        )
        for prompt in prompts
    ]
    responses = await asyncio.gather(*tasks)
    return responses


class OpenAIChatRuntime(Runtime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion models to perform the skill.

    Attributes:
        openai_model: OpenAI model name.
        openai_api_key: OpenAI API key. If not provided, will be taken from OPENAI_API_KEY environment variable.
        base_url: Can point to any implementation of the OpenAI API. Defaults to OpenAI's.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        splitter: Splitter to use for splitting input into multiple messages. Defaults to None.
        logprobs: Whether to include logprobs in the response. Defaults to False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    openai_model: str = Field(alias="model")
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"), alias="api_key"
    )
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 1000
    splitter: Optional[str] = None
    logprobs: Optional[bool] = False
    temperature: Optional[float] = 0.0

    @computed_field
    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

    def init_runtime(self) -> "Runtime":
        # check model availability
        try:
            self._client.models.retrieve(self.openai_model)
        except NotFoundError:
            raise ValueError(
                f'Requested model "{self.openai_model}" is not available in your OpenAI account.'
            )
        return self

    def execute(self, messages: List) -> Dict[str, str]:
        """
        Execute OpenAI request given list of messages in OpenAI API format
        """
        if self.verbose:
            print(f"OpenAI request: {messages}")

        completion = self._client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            logprobs=self.logprobs,
            temperature=self.temperature,
        )
        completion_text = completion.choices[0].message.content
        if self.logprobs:
            logprobs = [item.logprob for item in completion.choices[0].logprobs.content]
            mean_logprobs = sum(logprobs) / len(logprobs)
        else:
            mean_logprobs = None

        if self.verbose:
            print(f"OpenAI response: {completion_text}")
        return {'text': completion_text, 'logprobs': mean_logprobs}

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = False,
    ) -> Dict[str, str]:
        """
        Execute OpenAI request given record and templates for input, instructions and output.

        Args:
            record: Record to be used for input, instructions and output templates.
            input_template: Template for input message.
            instructions_template: Template for instructions message.
            output_template: Template for output message.
            extra_fields: Extra fields to be used in templates.
            field_schema: Field schema to be used for parsing templates.
            instructions_first: If True, instructions will be sent before input.

        Returns:
            Dict[str, str]: Output record.
        """

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        options = {}
        for field, schema in field_schema.items():
            if schema.get("type") == "array":
                options[field] = schema.get("items", {}).get("enum", [])

        output_fields = parse_template(
            partial_str_format(output_template, **extra_fields), include_texts=True
        )
        system_prompt = instructions_template
        user_prompt = input_template.format(**record, **extra_fields)
        messages = [{"role": "system", "content": system_prompt}]

        outputs = {}
        for output_field in output_fields:
            if output_field["type"] == "text":
                if user_prompt is not None:
                    user_prompt += f"\n{output_field['text']}"
                else:
                    user_prompt = output_field["text"]
            elif output_field["type"] == "var":
                name = output_field["text"]
                messages.append({"role": "user", "content": user_prompt})
                completion = self.execute(messages)
                if name in options:
                    completion_text = match_options(completion['text'], options[name])
                else:
                    completion_text = completion['text']
                outputs[name] = completion_text
                if self.logprobs:
                    outputs[f"{name}__logprobs"] = completion['logprobs']
                messages.append({"role": "assistant", "content": completion_text})
                user_prompt = None

        return outputs


class AsyncOpenAIChatRuntime(AsyncRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion models to perform the skill.
    It uses async calls to OpenAI API.

    Attributes:
        openai_model: OpenAI model name.
        openai_api_key: OpenAI API key. If not provided, will be taken from OPENAI_API_KEY environment variable.
        base_url: Can point to any implementation of the OpenAI API. Defaults to OpenAI's.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        temperature: Temperature for sampling, between 0 and 1. Higher values means the model will take more risks.
            Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            Defaults to 0.0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    openai_model: str = Field(alias="model")
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"), alias="api_key"
    )
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
    splitter: Optional[str] = None
    timeout: Optional[int] = 10

    @computed_field
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.openai_api_key,
            base_url=self.base_url,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.concurrent_clients,
                    max_keepalive_connections=self.concurrent_clients,
                ),
                timeout=self.timeout,
            ),
        )

    def init_runtime(self) -> "Runtime":
        # check model availability
        try:
            _client = OpenAI(api_key=self.openai_api_key)
            _client.models.retrieve(self.openai_model)
        except NotFoundError:
            raise ValueError(
                f'Requested model "{self.openai_model}" is not available in your OpenAI account.'
            )
        return self

    @field_validator("concurrency", mode="before")
    def check_concurrency(cls, value) -> int:
        value = value or -1
        if value < 1:
            raise NotImplementedError(
                "You must explicitly specify the number of concurrent clients for AsyncOpenAIChatRuntime. "
                "Set `AsyncOpenAIChatRuntime(concurrency=10, ...)` or any other positive integer. ")
        return value

    def _prepare_prompt(
        self,
        row,
        input_template: str,
        instructions_template: str,
        suffix: str,
        extra_fields: dict,
    ) -> Dict[str, str]:
        """Prepare input prompt for OpenAI API from the row of the dataframe"""
        return {
            "system": instructions_template,
            "user": input_template.format(**row, **extra_fields) + suffix,
        }

    async def batch_to_batch(
        self,
        batch: InternalDataFrame,
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> InternalDataFrame:
        """Execute batch of requests with async calls to OpenAI API"""

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        options = {}
        for field, schema in field_schema.items():
            if schema.get("type") == "array":
                options[field] = schema.get("items", {}).get("enum", [])

        output_fields = parse_template(
            partial_str_format(output_template, **extra_fields), include_texts=True
        )

        if len(output_fields) > 2:
            raise NotImplementedError("Only one output field is supported")

        suffix = ""
        outputs = []
        for output_field in output_fields:
            if output_field["type"] == "text":
                suffix += output_field["text"]

            elif output_field["type"] == "var":
                name = output_field["text"]
                # prepare prompts
                prompts = batch.apply(
                    lambda row: self._prepare_prompt(
                        row, input_template, instructions_template, suffix, extra_fields
                    ),
                    axis=1,
                ).tolist()

                responses = await async_concurrent_create_completion(
                    prompts=prompts,
                    client=self._client,
                    instruction_first=instructions_first,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    openai_model=self.openai_model,
                )

                # parse responses, optionally match it with options
                for prompt, response in zip(prompts, responses):
                    completion_text = response.pop("text")
                    if self.verbose:
                        if response["error"] is not None:
                            print_error(
                                f"Prompt: {prompt}\nOpenAI API error: {response}"
                            )
                        else:
                            print(
                                f"Prompt: {prompt}\nOpenAI API response: {completion_text}"
                            )
                    if name in options and completion_text is not None:
                        completion_text = match_options(completion_text, options[name])
                    # still technically possible to have a name collision here with the error, message, details fields
                    # `name in options` is only `True` for categorical variables, but is never `True` for freeform text generation
                    response[name] = completion_text
                    outputs.append(response)

        # TODO: note that this doesn't work for multiple output fields e.g. `Output {output1} and Output {output2}`
        output_df = InternalDataFrame(outputs)
        return output_df.set_index(batch.index)

    async def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, Any]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = True,
    ) -> Dict[str, str]:
        raise NotImplementedError("record_to_record is not implemented")


class OpenAIVisionRuntime(OpenAIChatRuntime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and vision models to perform the skill.
    Only compatible with OpenAI API version 1.0.0 or higher.
    """

    def record_to_record(
        self,
        record: Dict[str, str],
        input_template: str,
        instructions_template: str,
        output_template: str,
        extra_fields: Optional[Dict[str, str]] = None,
        field_schema: Optional[Dict] = None,
        instructions_first: bool = False,
    ) -> Dict[str, str]:
        """
        Execute OpenAI request given record and templates for input, instructions and output.

        Args:
            record: Record to be used for input, instructions and output templates.
            input_template: Template for input message.
            instructions_template: Template for instructions message.
            output_template: Template for output message.
            extra_fields: Extra fields to be used in templates.
            field_schema: Field jsonschema to be used for parsing templates.
                         Field schema must contain "format": "uri" for image fields. For example:
                            ```json
                            {
                                "image": {
                                    "type": "string",
                                    "format": "uri"
                                }
                            }
                            ```
            instructions_first: If True, instructions will be sent before input.
        """

        if not check_if_new_openai_version():
            raise NotImplementedError(
                f"{self.__class__.__name__} requires OpenAI API version 1.0.0 or higher."
            )

        extra_fields = extra_fields or {}
        field_schema = field_schema or {}

        output_fields = parse_template(
            partial_str_format(output_template, **extra_fields), include_texts=False
        )

        if len(output_fields) > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support multiple output fields. "
                f"Found: {output_fields}"
            )
        output_field = output_fields[0]
        output_field_name = output_field["text"]

        input_fields = parse_template(input_template)

        # split input template into text and image parts
        input_text = ""
        content = [
            {
                "type": "text",
                "text": instructions_template,
            }
        ]
        for field in input_fields:
            if field["type"] == "text":
                input_text += field["text"]
            elif field["type"] == "var":
                if field["text"] not in field_schema:
                    input_text += record[field["text"]]
                elif field_schema[field["text"]]["type"] == "string":
                    if field_schema[field["text"]].get("format") == "uri":
                        if input_text:
                            content.append({"type": "text", "text": input_text})
                            input_text = ""
                        content.append(
                            {"type": "image_url", "image_url": record[field["text"]]}
                        )
                    else:
                        input_text += record[field["text"]]
                else:
                    raise ValueError(
                        f'Unsupported field type: {field_schema[field["text"]]["type"]}'
                    )
        if input_text:
            content.append({"type": "text", "text": input_text})

        if self.verbose:
            print(f"**Prompt content**:\n{content}")

        completion = self._client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
        )

        completion_text = completion.choices[0].message.content
        return {output_field_name: completion_text}
