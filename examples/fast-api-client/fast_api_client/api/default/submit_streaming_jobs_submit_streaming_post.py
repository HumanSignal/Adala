from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.adala_response_job_created import AdalaResponseJobCreated
from ...models.http_validation_error import HTTPValidationError
from ...models.submit_streaming_request import SubmitStreamingRequest
from ...types import Response


def _get_kwargs(
    *,
    body: SubmitStreamingRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/jobs/submit-streaming",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AdalaResponseJobCreated, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AdalaResponseJobCreated.from_dict(response.json())

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
) -> Response[Union[AdalaResponseJobCreated, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SubmitStreamingRequest,
) -> Response[Union[AdalaResponseJobCreated, HTTPValidationError]]:
    """Submit Streaming

     Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.

    Args:
        body (SubmitStreamingRequest): Request model for submitting a streaming job.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseJobCreated, HTTPValidationError]]
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
    body: SubmitStreamingRequest,
) -> Optional[Union[AdalaResponseJobCreated, HTTPValidationError]]:
    """Submit Streaming

     Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.

    Args:
        body (SubmitStreamingRequest): Request model for submitting a streaming job.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseJobCreated, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SubmitStreamingRequest,
) -> Response[Union[AdalaResponseJobCreated, HTTPValidationError]]:
    """Submit Streaming

     Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.

    Args:
        body (SubmitStreamingRequest): Request model for submitting a streaming job.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseJobCreated, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: SubmitStreamingRequest,
) -> Optional[Union[AdalaResponseJobCreated, HTTPValidationError]]:
    """Submit Streaming

     Submit a request to execute task `request.task_name` in celery.

    Args:
        request (SubmitStreamingRequest): The request model for submitting a job.

    Returns:
        Response[JobCreated]: The response model for a job created.

    Args:
        body (SubmitStreamingRequest): Request model for submitting a streaming job.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseJobCreated, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
