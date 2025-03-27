import asyncio
import logging
import traceback
import litellm
from litellm import token_counter
from collections import defaultdict
from typing import Any, Dict, List, Type, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python
from litellm.types.utils import Usage
from litellm.utils import trim_messages
from tenacity import Retrying, AsyncRetrying
from instructor.exceptions import InstructorRetryException, IncompleteOutputException
from instructor.client import Instructor, AsyncInstructor
from adala.utils.parse import MessagesBuilder, MessageChunkType
from adala.utils.exceptions import ConstrainedGenerationError
from adala.utils.types import debug_time_it
from adala.utils.model_info_utils import match_model_provider_string, NoModelsFoundError
from litellm.exceptions import BadRequestError

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
    except Exception as e:
        logger.exception(f"Failed to get cost for model {model}", exc_info=e)
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
    e: Exception,
    messages: List[Dict[str, str]],
    model: str,
    retries,
    prompt_token_count: Optional[int] = None,
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
        if prompt_token_count is None:
            prompt_token_count = token_counter(model=model, messages=messages[:-1])
        if type(e).__name__ in {
            "APIError",
            "AuthenticationError",
            "APIConnectionError",
        }:
            prompt_tokens = 0
        else:
            prompt_tokens = n_attempts * prompt_token_count
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


def _ensure_messages_fit_in_context_window(
    messages: List[Dict[str, str]], model: str
) -> Tuple[List[Dict[str, str]], int]:
    """
    Ensure that the messages fit in the context window of the model.
    """
    token_count = token_counter(model=model, messages=messages)
    logger.debug(f"Prompt tokens count: {token_count}")

    if model in litellm.model_cost:
        # If we are able to identify the model context window, ensure the messages fit in it
        max_tokens = litellm.model_cost[model].get(
            "max_input_tokens", litellm.model_cost[model]["max_tokens"]
        )
        if token_count > max_tokens:
            logger.info(
                f"Prompt tokens count {token_count} exceeds max tokens {max_tokens} for model {model}. Trimming messages."
            )
            # TODO: in case it exceeds max tokens, content of the last message is truncated.
            # to improve this, we implement:
            # - UI-level warning for the user, use prediction_meta field for warnings as well as errors in future
            # - sequential aggregation instead of trimming
            # - potential v2 solution to downsample images instead of cutting them off (using quality=low instead of quality=auto in completion)
            return trim_messages(messages, model=model), token_count
    # in other cases, just return the original messages
    return messages, token_count


@debug_time_it
def run_instructor_with_messages(
    client: Instructor,
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str,
    canonical_model_provider_string: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[Retrying] = None,
    ensure_messages_fit_in_context_window: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a completion with an instructor client and handle errors appropriately.

    Args:
        client: The instructor client to use
        messages: The messages to send to the model
        response_model: The Pydantic model to validate the response against
        model: The model name to use
        canonical_model_provider_string: The canonical model provider string to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        ensure_messages_fit_in_context_window: Whether to ensure the messages fit in the context window (setting it to True will slow down the function)
        **kwargs: Additional arguments to pass to the completion call

    Returns:
        Dict containing the parsed response and usage information
    """
    try:
        prompt_token_count = None
        if ensure_messages_fit_in_context_window:
            messages, prompt_token_count = _ensure_messages_fit_in_context_window(
                messages, canonical_model_provider_string or model
            )

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
        dct, usage = handle_llm_exception(
            e, messages, model, retries, prompt_token_count=prompt_token_count
        )
        # With exceptions we don't have access to completion.model
        usage_model = model

    # Add usage data to the response (e.g. token counts, cost)
    dct.update(_get_usage_dict(usage, model=usage_model))

    return dct


@debug_time_it
async def arun_instructor_with_messages(
    client: AsyncInstructor,
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str,
    canonical_model_provider_string: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[AsyncRetrying] = None,
    ensure_messages_fit_in_context_window: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a completion with an instructor client and handle errors appropriately.

    Args:
        client: The instructor client to use
        messages: The messages to send to the model
        response_model: The Pydantic model to validate the response against
        model: The model name to use
        canonical_model_provider_string: The canonical model provider string to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        ensure_messages_fit_in_context_window: Whether to ensure the messages fit in the context window (setting it to True will slow down the function)
        **kwargs: Additional arguments to pass to the completion call

    Returns:
        Dict containing the parsed response and usage information
    """
    try:
        prompt_token_count = None
        if ensure_messages_fit_in_context_window:
            messages, prompt_token_count = _ensure_messages_fit_in_context_window(
                messages, canonical_model_provider_string or model
            )

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
        dct, usage = handle_llm_exception(
            e, messages, model, retries, prompt_token_count=prompt_token_count
        )
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
    canonical_model_provider_string: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[Retrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    split_into_chunks: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    ensure_messages_fit_in_context_window: bool = False,
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
        canonical_model_provider_string: The canonical model provider string to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        split_into_chunks: Whether to split the user prompt into chunks
        extra_fields: Additional fields to send to the model
        ensure_messages_fit_in_context_window: Whether to ensure the messages fit in the context window (setting it to True will slow down the function)
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
        canonical_model_provider_string,
        max_tokens,
        temperature,
        seed,
        retries,
        ensure_messages_fit_in_context_window=ensure_messages_fit_in_context_window,
        **kwargs,
    )


async def arun_instructor_with_payload(
    client: AsyncInstructor,
    payload: Dict[str, Any],
    user_prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    canonical_model_provider_string: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[AsyncRetrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    split_into_chunks: bool = False,
    ensure_messages_fit_in_context_window: bool = False,
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
        ensure_messages_fit_in_context_window: Whether to ensure the messages fit in the context window (setting it to True will slow down the function)
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
        canonical_model_provider_string,
        max_tokens,
        temperature,
        seed,
        retries,
        ensure_messages_fit_in_context_window=ensure_messages_fit_in_context_window,
        **kwargs,
    )


def run_instructor_with_payloads(
    client: Instructor,
    payloads: List[Dict[str, Any]],
    user_prompt_template: str,
    response_model: Type[BaseModel],
    model: str,
    canonical_model_provider_string: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[Retrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    split_into_chunks: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    ensure_messages_fit_in_context_window: bool = False,
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
        canonical_model_provider_string: The canonical model provider string to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        split_into_chunks: Whether to split the user prompt into chunks
        extra_fields: Additional fields to send to the model
        ensure_messages_fit_in_context_window: Whether to ensure the messages fit in the context window (setting it to True will slow down the function)
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
            canonical_model_provider_string,
            max_tokens,
            temperature,
            seed,
            retries,
            ensure_messages_fit_in_context_window=ensure_messages_fit_in_context_window,
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
    canonical_model_provider_string: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    retries: Optional[AsyncRetrying] = None,
    instructions_template: Optional[str] = None,
    instructions_first: bool = True,
    input_field_types: Optional[Dict[str, MessageChunkType]] = None,
    split_into_chunks: bool = False,
    extra_fields: Optional[Dict[str, Any]] = None,
    ensure_messages_fit_in_context_window: bool = False,
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
        canonical_model_provider_string: The canonical model provider string to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        seed: Integer seed to reduce nondeterminism
        retries: Retry policy to use
        instructions_template: The template to use for the instructions
        instructions_first: Whether to insert the instructions at the beginning of the message
        input_field_types: The types of the input fields
        split_into_chunks: Whether to split the user prompt into chunks
        extra_fields: Additional fields to send to the model
        ensure_messages_fit_in_context_window: Whether to ensure the messages fit in the context window (setting it to True will slow down the function)
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
            canonical_model_provider_string,
            max_tokens,
            temperature,
            seed,
            retries,
            ensure_messages_fit_in_context_window=ensure_messages_fit_in_context_window,
            **kwargs,
        )
        for payload in payloads
    ]

    return await asyncio.gather(*tasks)
