import os
import httpx
from loguru import logger
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

"""
This script is a simple HTTP proxy server that forwards requests to a target URL.
It requires the TARGET_URL environment variable to be set to the target URL.
It also requires the EXPECTED_HEADER environment variable to be set to the expected Authorization header value.

To install the dependencies, run the following command:
```
pip install fastapi httpx loguru
```

To run the server:
```
TARGET_URL=https://example.com EXPECTED_HEADER=secret uvicorn auth_proxy_server:app
```

This will forward all requests to `https://example.com` and check for the `Authorization` header to be equal to `secret`.
"""
app = FastAPI()

TARGET_URL = os.getenv('TARGET_URL')
EXPECTED_HEADER = os.getenv('EXPECTED_HEADER')


async def proxy_request(request: Request):
    # Check for authentication header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    if EXPECTED_HEADER and auth_header != EXPECTED_HEADER:
        raise HTTPException(status_code=403, detail=f"Invalid Authorization header."
                                                    f" Provided: {auth_header}. Required: {EXPECTED_HEADER}")

    # Prepare the URL for the proxied request
    path = request.url.path
    if request.url.query:
        path += f"?{request.url.query}"
    url = f"{TARGET_URL}{path}"

    # Prepare headers
    headers = dict(request.headers)
    headers["host"] = TARGET_URL.split("://")[1]

    logger.info(f"Forwarding request to {url}, headers: {headers}")

    # Create httpx client
    async with httpx.AsyncClient(timeout=60) as client:
        # Forward the request
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=await request.body()
        )

    # Stream the response back to the client
    return StreamingResponse(
        response.aiter_bytes(),
        status_code=response.status_code,
        headers=response.headers
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def catch_all(request: Request, path: str):
    return await proxy_request(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)