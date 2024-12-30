from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.adala_response_job_status_response import AdalaResponseJobStatusResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    job_id: Any,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": f"/jobs/{job_id}",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AdalaResponseJobStatusResponse.from_dict(response.json())

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
) -> Response[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    job_id: Any,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]:
    """Cancel Job

     Cancel a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse[status.CANCELED]

    Args:
        job_id (Any):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        job_id=job_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    job_id: Any,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]:
    """Cancel Job

     Cancel a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse[status.CANCELED]

    Args:
        job_id (Any):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseJobStatusResponse, HTTPValidationError]
    """

    return sync_detailed(
        job_id=job_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    job_id: Any,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]:
    """Cancel Job

     Cancel a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse[status.CANCELED]

    Args:
        job_id (Any):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        job_id=job_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    job_id: Any,
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[Union[AdalaResponseJobStatusResponse, HTTPValidationError]]:
    """Cancel Job

     Cancel a job.

    Args:
        job_id (str)

    Returns:
        JobStatusResponse[status.CANCELED]

    Args:
        job_id (Any):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseJobStatusResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            job_id=job_id,
            client=client,
        )
    ).parsed
