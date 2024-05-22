import os
import difflib
import asyncio
from rich import print

from typing import Optional, Dict, Any, List
from openai import OpenAI, NotFoundError, AsyncOpenAI
from pydantic import Field, computed_field, ConfigDict
from .base import Runtime, AsyncRuntime
from adala.utils.logs import print_error
from adala.utils.internal_data import InternalDataFrame, InternalSeries
from adala.utils.parse import parse_template, partial_str_format
from adala.utils.matching import match_options
from tenacity import retry, stop_after_attempt, wait_random_exponential
import httpx


DEFAULT_CREATE_COMPLETION_URL = "https://api.openai.com/v1/chat/completions"


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
            "error": False,
            "message": None,
            "details": None,
        }
    except Exception as e:
        # Handle other exceptions
        return {
            "text": None,
            "error": True,
            "message": "Unknown error",
            "details": str(e),
        }


async def async_concurrent_create_completion(
    prompts,
    client,
    instruction_first,
    openai_model,
    max_tokens,
    temperature,
):
    tasks = []
    for prompt in prompts:
        task = asyncio.ensure_future(
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
        tasks.append(task)
    responses = await asyncio.gather(*tasks)
    return responses


class OpenAIChatRuntime(Runtime):
    """
    Runtime that uses [OpenAI API](https://openai.com/) and chat completion models to perform the skill.

    Attributes:
        openai_model: OpenAI model name.
        openai_api_key: OpenAI API key. If not provided, will be taken from OPENAI_API_KEY environment variable.
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    openai_model: str = Field(alias="model")
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"), alias="api_key"
    )
    max_tokens: Optional[int] = 1000
    splitter: Optional[str] = None

    @computed_field
    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.openai_api_key)

    def init_runtime(self) -> "Runtime":
        # check model availability
        try:
            self._client.models.retrieve(self.openai_model)
        except NotFoundError:
            raise ValueError(
                f'Requested model "{self.openai_model}" is not available in your OpenAI account.'
            )
        return self

    def execute(self, messages: List):
        """
        Execute OpenAI request given list of messages in OpenAI API format
        """
        if self.verbose:
            print(f"OpenAI request: {messages}")

        completion = self._client.chat.completions.create(
            model=self.openai_model, messages=messages
        )
        completion_text = completion.choices[0].message.content

        if self.verbose:
            print(f"OpenAI response: {completion_text}")
        return completion_text

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
                completion_text = self.execute(messages)
                if name in options:
                    completion_text = match_options(completion_text, options[name])
                outputs[name] = completion_text
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
        max_tokens: Maximum number of tokens to generate. Defaults to 1000.
        temperature: Temperature for sampling, between 0 and 1. Higher values means the model will take more risks.
            Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            Defaults to 0.0.

        concurrent_clients: Number of concurrent clients to OpenAI API. More clients means more parallel requests, but
            also more money spent and more chances to hit the rate limit. Defaults to 10.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for @computed_field

    openai_model: str = Field(alias="model")
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"), alias="api_key"
    )
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.0
    splitter: Optional[str] = None
    concurrent_clients: Optional[int] = 10
    timeout: Optional[int] = 10

    @computed_field
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.openai_api_key,
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
                        # TODO check for key collision between name and "error", "message", "details" - or use reserved names for them
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
