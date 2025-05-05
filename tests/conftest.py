import pytest
import httpx
from unittest import mock

# from contextlib import asynccontextmanager
import pytest_asyncio
from fakeredis import FakeStrictRedis
from fastapi.testclient import TestClient
from server.app import _get_redis_conn


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "filter_query_parameters": ["api_key", "key"],
        "match_on": ("method", "scheme", "host", "port", "path", "query", "body"),
    }


def pytest_configure(config):
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
