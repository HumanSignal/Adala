import asyncio
import instructor
import litellm
import traceback
import multiprocessing as mp
from typing import Optional, Dict, Any, List, Type, Union
from pydantic import BaseModel, Field

instructor_client = instructor.from_litellm(litellm.completion)
async_instructor_client = instructor.from_litellm(litellm.acompletion)


class LLMResponse(BaseModel):
    data: Dict = Field(default_factory=dict)
    adala_error: bool = Field(default=False, serialization_alias='_adala_error')
    adala_message: str = Field(default=None, serialization_alias='_adala_message')
    adala_details: str = Field(default=None, serialization_alias='_adala_details')
    # `text` is deprecated, use `data` instead
    text: str = Field(default=None, deprecated=True)


def get_messages(user_prompt: str, system_prompt: Optional[str] = None, instruction_first: bool = True):
    messages = [{'role': 'user', 'content': user_prompt}]
    if system_prompt:
        if instruction_first:
            messages.insert(0, {'role': 'system', 'content': system_prompt})
        else:
            messages[0]['content'] += system_prompt
    return messages


async def async_get_llm_response(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = 'gpt-4o-mini',
    instruction_first: Optional[bool] = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.0,
    timeout: Optional[Union[float, int]] = None,
) -> Dict[str, Any]:
    """
    Async version of create_completion function with error handling and session timeout.

    Args:
        model: model name. Refer to litellm supported models for how to pass
               this: https://litellm.vercel.app/docs/providers
        user_prompt: User prompt.
        system_prompt: System prompt.
        api_key: API key, optional. If provided, will be used to authenticate
                 with the provider of your specified model.
        instruction_first: Whether to put instructions first.
        response_model: Pydantic model to constrain the LLM generated response. If not provided, the raw completion text will be returned.  # noqa
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.
        timeout: Timeout in seconds.

    Returns:
        Dict[str, Any]: OpenAI response or error message.
    """
    messages = get_messages(user_prompt, system_prompt, instruction_first)
    response = LLMResponse()

    if response_model is None:
        # unconstrained generation - return raw completion text and store it in `data` field: {"text": completion_text}
        try:
            completion = await litellm.acompletion(
                model=model,
                api_key=api_key,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            completion_text = completion.choices[0].message.content
            response.data['text'] = completion_text
            # TODO: `text` is deprecated - remove it and use `data` instead
            response.text = completion_text
        except Exception as e:
            response.adala_error = True
            response.adala_message = type(e).__name__
            # create a traceback string
            response.adala_details = traceback.format_exc()
        finally:
            return response.model_dump(by_alias=True)

    # constrained generation branch - use `response_model` to constrain the LLM response
    try:
        instructor_response, completion = await async_instructor_client.chat.completions.create_with_completion(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            timeout=timeout,
        )
        response.data = instructor_response.model_dump(by_alias=True)
        # TODO: `text` is deprecated - remove it and use `data` instead
        response.text = completion.choices[0].message.content
    except Exception as e:
        response.adala_error = True
        response.adala_message = type(e).__name__
        response.adala_details = traceback.format_exc()
    finally:
        return response.model_dump(by_alias=True)


async def parallel_async_get_llm_response(
    user_prompts: List[str],
    system_prompt: Optional[str] = None,
    model: Optional[str] = 'gpt-4o-mini',
    instruction_first: Optional[bool] = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.0,
    timeout: Optional[Union[float, int]] = None,
):
    tasks = [
        asyncio.ensure_future(
            async_get_llm_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                api_key=api_key,
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
    user_prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = 'gpt-4o-mini',
    instruction_first: Optional[bool] = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.0,
    timeout: Optional[Union[float, int]] = None,
) -> Dict[str, Any]:

    # format messages
    messages = get_messages(user_prompt, system_prompt, instruction_first)
    response = LLMResponse()

    if response_model is None:
        # unconstrained generation - return raw completion text and store it in `data` field: {"text": completion_text}
        # TODO: this branch can be considered as deprecated at some point, as we always want to run LLM constrained by pydantic model  # noqa
        try:
            completion = litellm.completion(
                model=model,
                api_key=api_key,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            completion_text = completion.choices[0].message.content
            response.data['text'] = completion_text
            # TODO: `text` is deprecated - remove it and use `data` instead
            response.text = completion_text
        except Exception as e:
            response.adala_error = True
            response.adala_message = type(e).__name__
            # create a traceback string
            response.adala_details = traceback.format_exc()
        finally:
            return response.model_dump(by_alias=True)

    # constrained generation branch - use `response_model` to constrain the LLM response
    try:
        instructor_response, completion = instructor_client.chat.completions.create_with_completion(
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            response_model=response_model,
        )
        response.data = instructor_response.model_dump(by_alias=True)
        # TODO: `text` is deprecated - remove it and use `data` instead
        response.text = completion.choices[0].message.content
    except Exception as e:
        response.adala_error = True
        response.adala_message = type(e).__name__
        response.adala_details = traceback.format_exc()
    finally:
        return response.model_dump(by_alias=True)


def parallel_get_llm_response(
    user_prompts: List[str],
    system_prompt: Optional[str] = None,
    model: Optional[str] = 'gpt-4o-mini',
    instruction_first: Optional[bool] = True,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = 1000,
    temperature: Optional[float] = 0.0
):
    pool = mp.Pool(mp.cpu_count())
    responses = pool.starmap(
        get_llm_response,
        [
            (
                user_prompt,
                system_prompt,
                model,
                instruction_first,
                response_model,
                api_key,
                max_tokens,
                temperature
            )
            for user_prompt in user_prompts
        ]
    )
    pool.close()
    pool.join()
    return responses
