import asyncio
import instructor
import litellm
import traceback
import multiprocessing as mp
from typing import Optional, Dict, List, Type, Union
from pydantic import BaseModel, Field

instructor_client = instructor.from_litellm(litellm.completion)
async_instructor_client = instructor.from_litellm(litellm.acompletion)


class LLMResponse(BaseModel):
    """
    Base class for LLM response.
    """


class ConstrainedLLMResponse(LLMResponse):
    """
    LLM response from constrained generation.
    `data` object contains fields required by the response model.
    """
    data: Dict = Field(default_factory=dict)


class UnconstrainedLLMResponse(LLMResponse):
    """
    LLM response from unconstrained generation.
    `text` field contains raw completion text.
    """
    text: str = Field(default=None)


class ErrorLLMResponse(LLMResponse):
    """
    LLM response in case of error.
    """
    adala_error: bool = Field(default=True, serialization_alias='_adala_error')
    adala_message: str = Field(default=None, serialization_alias='_adala_message')
    adala_details: str = Field(default=None, serialization_alias='_adala_details')


def get_messages(user_prompt: str, system_prompt: Optional[str] = None, instruction_first: bool = True):
    messages = [{'role': 'user', 'content': user_prompt}]
    if system_prompt:
        if instruction_first:
            messages.insert(0, {'role': 'system', 'content': system_prompt})
        else:
            messages[0]['content'] += system_prompt
    return messages


async def async_get_llm_response(
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: str = 'gpt-4o-mini',
    instruction_first: bool = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    timeout: Optional[Union[float, int]] = None,
) -> LLMResponse:
    """
    Async version of create_completion function with error handling and session timeout.

    Args:
        model: model name. Refer to litellm supported models for how to pass
               this: https://litellm.vercel.app/docs/providers
        user_prompt: User prompt.
        system_prompt: System prompt.
        messages: List of messages to be sent to the model. If provided, `user_prompt`, `system_prompt` and `instruction_first` will be ignored.
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        base_url (Optional[str]): Base URL, optional. If provided, will be used to talk to an OpenAI-compatible API provider besides OpenAI.
        api_version (Optional[str]): API version, optional except for Azure.
        instruction_first: Whether to put instructions first.
        response_model: Pydantic model to constrain the LLM generated response. If not provided, the raw completion text will be returned.  # noqa
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.
        timeout: Timeout in seconds.

    Returns:
        LLMResponse: OpenAI response or error message.
    """

    if not user_prompt and not messages:
        raise ValueError("You must provide either `user_prompt` or `messages`.")

    if not messages:
        # get messages from user_prompt and system_prompt
        messages = get_messages(user_prompt, system_prompt, instruction_first)

    if response_model is None:
        # unconstrained generation - return raw completion text and store it in `data` field: {"text": completion_text}
        try:
            completion = await litellm.acompletion(
                model=model,
                api_key=api_key,
                api_base=base_url,
                api_version=api_version,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            completion_text = completion.choices[0].message.content
            return UnconstrainedLLMResponse(text=completion_text)
        except Exception as e:
            return ErrorLLMResponse(adala_message=type(e).__name__, adala_details=traceback.format_exc())

    # constrained generation branch - use `response_model` to constrain the LLM response
    try:
        instructor_response, completion = await async_instructor_client.chat.completions.create_with_completion(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            timeout=timeout,
        )
        return ConstrainedLLMResponse(data=instructor_response.model_dump(by_alias=True))
    except Exception as e:
        return ErrorLLMResponse(adala_message=type(e).__name__, adala_details=traceback.format_exc())


async def parallel_async_get_llm_response(
    user_prompts: List[str],
    system_prompt: Optional[str] = None,
    model: str = 'gpt-4o-mini',
    instruction_first: bool = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    timeout: Optional[Union[float, int]] = None,
):
    tasks = [
        asyncio.ensure_future(
            async_get_llm_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                api_version=api_version,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                instruction_first=instruction_first,
                response_model=response_model,
            )
        )
        for user_prompt in user_prompts
    ]
    responses = await asyncio.gather(*tasks)
    return responses


def get_llm_response(
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: str = 'gpt-4o-mini',
    instruction_first: bool = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    timeout: Optional[Union[float, int]] = None,
) -> LLMResponse:

    """
    Synchronous version of create_completion function with error handling and session timeout.

    Args:
        user_prompt (Optional[str]): User prompt.
        system_prompt (Optional[str]): System prompt.
        messages (Optional[List[Dict[str, str]]]): List of messages to be sent to the model. If provided, `user_prompt`, `system_prompt` and `instruction_first` will be ignored.
        model (Optional[str]): Model name. Refer to litellm supported models for how to pass this: https://litellm.vercel.app/docs/providers
        instruction_first (Optional[bool]): Whether to put instructions first.
        response_model (Optional[Type[BaseModel]]): Pydantic model to constrain the LLM generated response. If not provided, the raw completion text will be returned.
        api_key (Optional[str]): API key, optional. If provided, will be used to authenticate with the provider of your specified model.
        base_url (Optional[str]): Base URL, optional. If provided, will be used to talk to an OpenAI-compatible API provider besides OpenAI.
        api_version (Optional[str]): API version, optional except for Azure.
        max_tokens (Optional[int]): Maximum tokens to generate.
        temperature (Optional[float]): Temperature for sampling.
        timeout (Optional[Union[float, int]]): Timeout in seconds.

    Returns:
        Dict[str, Any]: OpenAI response or error message.
    """

    if not user_prompt and not messages:
        raise ValueError("You must provide either `user_prompt` or `messages`.")

    if not messages:
        # get messages from user_prompt and system_prompt
        messages = get_messages(user_prompt, system_prompt, instruction_first)

    if response_model is None:
        # unconstrained generation - return raw completion text and store it in `data` field: {"text": completion_text}
        # TODO: this branch can be considered as deprecated at some point, as we always want to run LLM constrained by pydantic model  # noqa
        try:
            completion = litellm.completion(
                model=model,
                api_key=api_key,
                api_base=base_url,
                api_version=api_version,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            completion_text = completion.choices[0].message.content
            return UnconstrainedLLMResponse(text=completion_text)
        except Exception as e:
            return ErrorLLMResponse(adala_message=type(e).__name__, adala_details=traceback.format_exc())

    # constrained generation branch - use `response_model` to constrain the LLM response
    try:
        instructor_response, completion = instructor_client.chat.completions.create_with_completion(
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            response_model=response_model,
        )
        return ConstrainedLLMResponse(data=instructor_response.model_dump(by_alias=True))
    except Exception as e:
        return ErrorLLMResponse(adala_message=type(e).__name__, adala_details=traceback.format_exc())


def parallel_get_llm_response(
    user_prompts: List[str],
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: str = 'gpt-4o-mini',
    instruction_first: bool = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    timeout: Optional[Union[float, int]] = None,
):
    pool = mp.Pool(mp.cpu_count())
    responses = pool.starmap(
        get_llm_response,
        [
            (
                user_prompt,
                system_prompt,
                messages,
                model,
                instruction_first,
                response_model,
                api_key,
                max_tokens,
                temperature,
                timeout
            )
            for user_prompt in user_prompts
        ]
    )
    pool.close()
    pool.join()
    return responses
