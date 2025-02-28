import asyncio
import logging
import traceback
import litellm
from litellm import token_counter
from collections import defaultdict
from typing import Any, Dict, List, Type, Optional
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python
from litellm.types.utils import Usage
from tenacity import Retrying, AsyncRetrying
from instructor.exceptions import InstructorRetryException, IncompleteOutputException
from instructor.client import Instructor, AsyncInstructor
from adala.utils.parse import MessagesBuilder, MessageChunkType

logger = logging.getLogger(__name__)


def _get_usage_dict(usage: Usage, model: str) -> Dict:
    data = dict()
    data["_prompt_tokens"] = usage.prompt_tokens

    # will not exist if there is no completion
    # sometimes the response will have a CompletionUsage instead of a Usage, which doesn't have a .get() method
    # data["_completion_tokens"] = usage.get("completion_tokens", 0)
    try:
        data["_completion_tokens"] = usage.completion_tokens
    except AttributeError:
        data["_completion_tokens"] = 0

    # can't use litellm.completion_cost bc it only takes the most recent completion, and .usage is summed over retries
    # TODO make sure this is calculated correctly after we turn on caching
    # litellm will register the cost of an azure model on first successful completion. If there hasn't been a successful completion, the model will not be registered
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model, data["_prompt_tokens"], data["_completion_tokens"]
        )
        data["_prompt_cost_usd"] = prompt_cost
        data["_completion_cost_usd"] = completion_cost
        data["_total_cost_usd"] = prompt_cost + completion_cost
    except:
        logger.exception(f"Failed to get cost for model {model}")
        data["_prompt_cost_usd"] = None
        data["_completion_cost_usd"] = None
        data["_total_cost_usd"] = None
    return data


def _format_error_dict(e: Exception) -> dict:
    error_message = type(e).__name__
    error_details = str(e)
    # TODO change this format?
    error_dct = {
        "_adala_error": True,
        "_adala_message": error_message,
        "_adala_details": error_details,
    }
    return error_dct


def _log_llm_exception(e) -> dict:
    dct = _format_error_dict(e)
    base_error = f"Inference error {dct['_adala_message']}"
    tb = "".join(
        traceback.format_exception(e)
    )  # format_exception return list of strings ending in new lines
    logger.error(f"{base_error}\nTraceback:\n{tb}")
    return dct


def handle_llm_exception(
    e: Exception, messages: List[Dict[str, str]], model: str, retries
) -> tuple[Dict, Usage]:
    """Handle exceptions from LLM calls and return standardized error dict and usage stats.

    Args:
        e: The caught exception
        messages: The messages that were sent to the LLM
        model: The model name
        retries: The retry policy object

    Returns:
        Tuple of (error_dict, usage_stats)
    """
    logger.debug(f"LLM Exception: {e}\nTraceback:\n{traceback.format_exc()}")
    if isinstance(e, IncompleteOutputException):
        usage = e.total_usage
    elif isinstance(e, InstructorRetryException):
        usage = e.total_usage
        # get root cause error from retries
        e = e.__cause__.last_attempt.exception()
    else:
        # Approximate usage for other errors
        # usage = e.total_usage
        # not available here, so have to approximate by hand, assuming the same error occurred each time
        n_attempts = retries.stop.max_attempt_number
        # Note that the default model used in token_counter is gpt-3.5-turbo as of now - if model passed in
        # does not match a mapped model, falls back to default
        prompt_tokens = n_attempts * litellm.token_counter(
            model=model, messages=messages[:-1]
        )  # response is appended as the last message
        # TODO a pydantic validation error may be appended as the last message, don't know how to get the raw response in this case
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        )
        # Catch case where the model does not return a properly formatted out
        # AttributeError is an instructor bug: https://github.com/instructor-ai/instructor/pull/1103
        # > AttributeError: 'NoneType' object has no attribute '_raw_response'
        if type(e).__name__ in {"ValidationError", "AttributeError"}:
            logger.error(f"Converting error to ConstrainedGenerationError: {str(e)}")
            e = ConstrainedGenerationError()

        # the only other instructor error that would be thrown is IncompleteOutputException due to max_tokens reached

    return _log_llm_exception(e), usage


def run_instructor_with_messages(
    client: Instructor,
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[Retrying] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a completion with an instructor client and handle errors appropriately.

    Args:
        client: The instructor client to use
        messages: The messages to send to the model
        response_model: The Pydantic model to validate the response against
        model: The model name to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        **kwargs: Additional arguments to pass to the completion call

    Returns:
        Dict containing the parsed response and usage information
    """
    try:
        # returns a pydantic model and completion info
        response, completion = client.chat.completions.create_with_completion(
            messages=messages,
            response_model=response_model,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            max_retries=retries,
            **kwargs,
        )
        usage = completion.usage
        dct = to_jsonable_python(response)
        # With successful completions we can get canonical model name
        usage_model = completion.model

    except Exception as e:
        dct, usage = handle_llm_exception(e, messages, model, retries)
        # With exceptions we don't have access to completion.model
        usage_model = model

    # Add usage data to the response (e.g. token counts, cost)
    dct.update(_get_usage_dict(usage, model=usage_model))

    return dct


async def arun_instructor_with_messages(
    client: AsyncInstructor,
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[AsyncRetrying] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a completion with an instructor client and handle errors appropriately.

    Args:
        client: The instructor client to use
        messages: The messages to send to the model
        response_model: The Pydantic model to validate the response against
        model: The model name to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        **kwargs: Additional arguments to pass to the completion call

    Returns:
        Dict containing the parsed response and usage information
    """
    try:
        # returns a pydantic model and completion info
        response, completion = await client.chat.completions.create_with_completion(
            messages=messages,
            response_model=response_model,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            max_retries=retries,
            **kwargs,
        )
        usage = completion.usage
        dct = to_jsonable_python(response)
        # With successful completions we can get canonical model name
        usage_model = completion.model

    except Exception as e:
        dct, usage = handle_llm_exception(e, messages, model, retries)
        # With exceptions we don't have access to completion.model
        usage_model = model

    # Add usage data to the response (e.g. token counts, cost)
    dct.update(_get_usage_dict(usage, model=usage_model))

    return dct


def run_instructor_with_payload(
    client: Instructor,
    payload: Dict[str, Any],
    user_prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[Retrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    split_into_chunks: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a completion with an instructor client and handle errors appropriately.

    Args:
        client: The instructor client to use
        payload: The data payload to send to the model
        user_prompt_template: The template to use for the user prompt
        response_model: The Pydantic model to validate the response against
        model: The model name to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        split_into_chunks: Whether to split the user prompt into chunks
        extra_fields: Additional fields to send to the model
        **kwargs: Additional arguments to pass to the completion call

    Returns:
        Dict containing the parsed response and usage information
    """

    messages_builder = MessagesBuilder(
        user_prompt_template=user_prompt_template,
        system_prompt=instructions_template,
        instructions_first=instructions_first,
        input_field_types=input_field_types,
        extra_fields=extra_fields,
        split_into_chunks=split_into_chunks,
    )

    messages = messages_builder.get_messages(payload)
    return run_instructor_with_messages(
        client,
        messages,
        response_model,
        model,
        max_tokens,
        temperature,
        seed,
        retries,
        **kwargs,
    )


async def arun_instructor_with_payload(
    client: AsyncInstructor,
    payload: Dict[str, Any],
    user_prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[AsyncRetrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    split_into_chunks: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a completion with an instructor client and handle errors appropriately.

    Args:
        client: The instructor client to use
        payload: The data payload to send to the model
        user_prompt_template: The template to use for the user prompt
        response_model: The Pydantic model to validate the response against
        model: The model name to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        extra_fields: Additional fields to send to the model
        split_into_chunks: Whether to split the user prompt into chunks
        **kwargs: Additional arguments to pass to the completion call

    Returns:
        Dict containing the parsed response and usage information
    """

    messages_builder = MessagesBuilder(
        user_prompt_template=user_prompt_template,
        system_prompt=instructions_template,
        instructions_first=instructions_first,
        input_field_types=input_field_types,
        extra_fields=extra_fields,
        split_into_chunks=split_into_chunks,
    )

    messages = messages_builder.get_messages(payload)
    return await arun_instructor_with_messages(
        client,
        messages,
        response_model,
        model,
        max_tokens,
        temperature,
        seed,
        retries,
        **kwargs,
    )


def run_instructor_with_payloads(
    client: Instructor,
    payloads: List[Dict[str, Any]],
    user_prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    retries: Retrying,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    split_into_chunks: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Run completions with an instructor client for multiple payloads and handle errors appropriately.
    Synchronous version of arun_instructor_completions.

    Args:
        client: The instructor client to use
        payloads: List of data payloads to send to the model
        user_prompt_template: The template to use for the user prompt
        response_model: The Pydantic model to validate the responses against
        model: The model name to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        split_into_chunks: Whether to split the user prompt into chunks
        extra_fields: Additional fields to send to the model
        **kwargs: Additional arguments to pass to the completion calls

    Returns:
        List of dicts containing the parsed responses and usage information
    """
    messages_builder = MessagesBuilder(
        user_prompt_template=user_prompt_template,
        system_prompt=instructions_template,
        instruction_first=instructions_first,
        input_field_types=input_field_types,
        extra_fields=extra_fields,
        split_into_chunks=split_into_chunks,
    )

    results = []
    for payload in payloads:
        messages = messages_builder.get_messages(payload)
        result = run_instructor_with_messages(
            client,
            messages,
            response_model,
            model,
            max_tokens,
            temperature,
            seed,
            retries,
            **kwargs,
        )
        results.append(result)

    return results


async def arun_instructor_with_payloads(
    client: AsyncInstructor,
    payloads: List[Dict[str, Any]],
    user_prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[AsyncRetrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    split_into_chunks: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Run completions with an instructor client for multiple payloads and handle errors appropriately.

    Args:
        client: The instructor client to use
        payloads: List of data payloads to send to the model
        user_prompt_template: The template to use for the user prompt
        response_model: The Pydantic model to validate the responses against
        model: The model name to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        split_into_chunks: Whether to split the user prompt into chunks
        extra_fields: Additional fields to send to the model
        **kwargs: Additional arguments to pass to the completion calls

    Returns:
        List of dicts containing the parsed responses and usage information
    """

    messages_builder = MessagesBuilder(
        user_prompt_template=user_prompt_template,
        system_prompt=instructions_template,
        instruction_first=instructions_first,
        input_field_types=input_field_types,
        extra_fields=extra_fields,
        split_into_chunks=split_into_chunks,
    )

    tasks = [
        arun_instructor_with_messages(
            client,
            messages_builder.get_messages(payload),
            response_model,
            model,
            max_tokens,
            temperature,
            seed,
            retries,
            **kwargs,
        )
        for payload in payloads
    ]

    return await asyncio.gather(*tasks)
