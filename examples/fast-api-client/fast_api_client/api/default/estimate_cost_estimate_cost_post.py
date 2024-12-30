from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.adala_response_cost_estimate import AdalaResponseCostEstimate
from ...models.cost_estimate_request import CostEstimateRequest
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: CostEstimateRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/estimate-cost",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AdalaResponseCostEstimate, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AdalaResponseCostEstimate.from_dict(response.json())

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
) -> Response[Union[AdalaResponseCostEstimate, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CostEstimateRequest,
) -> Response[Union[AdalaResponseCostEstimate, HTTPValidationError]]:
    """Estimate Cost

     Estimates what it would cost to run inference on the batch of data in
    `request` (using the run params from `request`)

    Args:
        request (CostEstimateRequest): Specification for the inference run to
            make an estimate for, includes:
                agent (adala.agent.Agent): The agent definition, used to get the model
                    and any other params necessary to estimate cost
                prompt (str): The prompt template that will be used for each task
                substitutions (List[Dict]): Mappings to substitute (simply using str.format)

    Returns:
        Response[CostEstimate]: The cost estimate, including the prompt/completion/total costs (in USD)

    Args:
        body (CostEstimateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseCostEstimate, HTTPValidationError]]
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
    body: CostEstimateRequest,
) -> Optional[Union[AdalaResponseCostEstimate, HTTPValidationError]]:
    """Estimate Cost

     Estimates what it would cost to run inference on the batch of data in
    `request` (using the run params from `request`)

    Args:
        request (CostEstimateRequest): Specification for the inference run to
            make an estimate for, includes:
                agent (adala.agent.Agent): The agent definition, used to get the model
                    and any other params necessary to estimate cost
                prompt (str): The prompt template that will be used for each task
                substitutions (List[Dict]): Mappings to substitute (simply using str.format)

    Returns:
        Response[CostEstimate]: The cost estimate, including the prompt/completion/total costs (in USD)

    Args:
        body (CostEstimateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseCostEstimate, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CostEstimateRequest,
) -> Response[Union[AdalaResponseCostEstimate, HTTPValidationError]]:
    """Estimate Cost

     Estimates what it would cost to run inference on the batch of data in
    `request` (using the run params from `request`)

    Args:
        request (CostEstimateRequest): Specification for the inference run to
            make an estimate for, includes:
                agent (adala.agent.Agent): The agent definition, used to get the model
                    and any other params necessary to estimate cost
                prompt (str): The prompt template that will be used for each task
                substitutions (List[Dict]): Mappings to substitute (simply using str.format)

    Returns:
        Response[CostEstimate]: The cost estimate, including the prompt/completion/total costs (in USD)

    Args:
        body (CostEstimateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AdalaResponseCostEstimate, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: CostEstimateRequest,
) -> Optional[Union[AdalaResponseCostEstimate, HTTPValidationError]]:
    """Estimate Cost

     Estimates what it would cost to run inference on the batch of data in
    `request` (using the run params from `request`)

    Args:
        request (CostEstimateRequest): Specification for the inference run to
            make an estimate for, includes:
                agent (adala.agent.Agent): The agent definition, used to get the model
                    and any other params necessary to estimate cost
                prompt (str): The prompt template that will be used for each task
                substitutions (List[Dict]): Mappings to substitute (simply using str.format)

    Returns:
        Response[CostEstimate]: The cost estimate, including the prompt/completion/total costs (in USD)

    Args:
        body (CostEstimateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AdalaResponseCostEstimate, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
