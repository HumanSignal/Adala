from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.adala_response_improved_prompt_response import AdalaResponseImprovedPromptResponse
from ...models.http_validation_error import HTTPValidationError
from ...models.improved_prompt_request import ImprovedPromptRequest
from ...types import Response


def _get_kwargs(
    *,
    body: ImprovedPromptRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/improved-prompt",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AdalaResponseImprovedPromptResponse.from_dict(response.json())

        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ImprovedPromptRequest,
) -> Response[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]:
    """Improved Prompt

     Improve a given prompt using the specified model and variables.

    Args:
        request (ImprovedPromptRequest): The request model for improving a prompt.

    Returns:
        Response: Response model for prompt improvement skill

    Args:
        body (ImprovedPromptRequest): Request model for improving a prompt.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ImprovedPromptRequest,
) -> Optional[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]:
    """Improved Prompt

     Improve a given prompt using the specified model and variables.

    Args:
        request (ImprovedPromptRequest): The request model for improving a prompt.

    Returns:
        Response: Response model for prompt improvement skill

    Args:
        body (ImprovedPromptRequest): Request model for improving a prompt.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ImprovedPromptRequest,
) -> Response[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]:
    """Improved Prompt

     Improve a given prompt using the specified model and variables.

    Args:
        request (ImprovedPromptRequest): The request model for improving a prompt.

    Returns:
        Response: Response model for prompt improvement skill

    Args:
        body (ImprovedPromptRequest): Request model for improving a prompt.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ImprovedPromptRequest,
) -> Optional[Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]]:
    """Improved Prompt

     Improve a given prompt using the specified model and variables.

    Args:
        request (ImprovedPromptRequest): The request model for improving a prompt.

    Returns:
        Response: Response model for prompt improvement skill

    Args:
        body (ImprovedPromptRequest): Request model for improving a prompt.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseImprovedPromptResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
