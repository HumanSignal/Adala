import pytest
import httpx
from unittest import mock
import os
from contextlib import suppress

# from contextlib import asynccontextmanager
import pytest_asyncio
from fakeredis import FakeStrictRedis
from fastapi.testclient import TestClient
from server.app import _get_redis_conn
from litellm import close_litellm_async_clients
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "filter_query_parameters": ["api_key", "key", "api-version"],
        # Dependency upgrades (e.g. OpenAI/LiteLLM/Instructor) frequently change request
        # bodies in non-semantic ways; matching on body makes existing cassettes brittle.
        "match_on": ("method", "scheme", "host", "port", "path", "query"),
    }


def pytest_configure(config):
    # Newer OpenAI / LiteLLM versions error early if no API key is present,
    # even when requests are replayed from VCR cassettes. Provide harmless defaults.
    os.environ.setdefault("OPENAI_API_KEY", "test")
    os.environ.setdefault("AZURE_OPENAI_API_KEY", "test")
    os.environ.setdefault("GOOGLE_API_KEY", "test")
    os.environ.setdefault("GEMINI_API_KEY", "test")

    config.addinivalue_line("markers", "use_openai: mark test as requiring OpenAI key")
    config.addinivalue_line(
        "markers", "use_azure: mark test as requiring Azure OpenAI key"
    )
    config.addinivalue_line(
        "markers", "use_server: mark test as requiring running adala server"
    )
    config.addinivalue_line(
        "addopts", "-m 'not (use_openai or use_azure or use_server)'"
    )


@pytest.fixture
def client():
    from server.app import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_client():
    from server.app import app

    async with httpx.AsyncClient(
        timeout=10,
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost:30001",
    ) as client:
        yield client


@pytest.fixture
def redis_mock(client):
    """
    only works with sync client, not async client
    """
    fake_redis = FakeStrictRedis()
    with mock.patch.dict(
        client.app.dependency_overrides, {_get_redis_conn: lambda: fake_redis}
    ):
        yield fake_redis


@pytest_asyncio.fixture(autouse=True)
async def cleanup_litellm_logging_worker():
    """
    Keep LiteLLM's global logging worker from leaking tasks across pytest event loops.
    https://github.com/BerriAI/litellm/issues/14521
    """
    yield

    # Flush first, then unbind queue from the current loop, then stop worker.
    with suppress(Exception):
        await GLOBAL_LOGGING_WORKER.flush()
    GLOBAL_LOGGING_WORKER._queue = None
    with suppress(Exception):
        await GLOBAL_LOGGING_WORKER.stop()
    with suppress(Exception):
        await close_litellm_async_clients()
