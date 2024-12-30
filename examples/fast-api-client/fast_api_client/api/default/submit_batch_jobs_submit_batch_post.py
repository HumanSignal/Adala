from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.adala_response import AdalaResponse
from ...models.batch_data import BatchData
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: BatchData,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/jobs/submit-batch",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AdalaResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AdalaResponse.from_dict(response.json())

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
) -> Response[Union[AdalaResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchData,
) -> Response[Union[AdalaResponse, HTTPValidationError]]:
    """Submit Batch

     Submits a batch of data to an existing streaming job.
    Will push the batch of data into Kafka in a topic specific to the job ID

    Args:
        batch (BatchData): The data to push to Kafka queue to be processed by agent.arun()

    Returns:
        Response: Generic response indicating status of request

    Args:
        body (BatchData): Model for a batch of data submitted to a streaming job

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponse, HTTPValidationError]]
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
    body: BatchData,
) -> Optional[Union[AdalaResponse, HTTPValidationError]]:
    """Submit Batch

     Submits a batch of data to an existing streaming job.
    Will push the batch of data into Kafka in a topic specific to the job ID

    Args:
        batch (BatchData): The data to push to Kafka queue to be processed by agent.arun()

    Returns:
        Response: Generic response indicating status of request

    Args:
        body (BatchData): Model for a batch of data submitted to a streaming job

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchData,
) -> Response[Union[AdalaResponse, HTTPValidationError]]:
    """Submit Batch

     Submits a batch of data to an existing streaming job.
    Will push the batch of data into Kafka in a topic specific to the job ID

    Args:
        batch (BatchData): The data to push to Kafka queue to be processed by agent.arun()

    Returns:
        Response: Generic response indicating status of request

    Args:
        body (BatchData): Model for a batch of data submitted to a streaming job

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: BatchData,
) -> Optional[Union[AdalaResponse, HTTPValidationError]]:
    """Submit Batch

     Submits a batch of data to an existing streaming job.
    Will push the batch of data into Kafka in a topic specific to the job ID

    Args:
        batch (BatchData): The data to push to Kafka queue to be processed by agent.arun()

    Returns:
        Response: Generic response indicating status of request

    Args:
        body (BatchData): Model for a batch of data submitted to a streaming job

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
