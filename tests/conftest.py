import pytest
import httpx
from unittest import mock

# from contextlib import asynccontextmanager
import pytest_asyncio
from fakeredis import FakeStrictRedis
from fastapi.testclient import TestClient
from server.app import _get_redis_conn
from server.utils import Settings
import os



@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}


def pytest_addoption(parser):
    parser.addoption(
        "--use-openai",
        action="store_true",
        default=False,
        help="run tests that require OpenAI key",
    )
    parser.addoption(
        "--use-server",
        action="store_true",
        default=False,
        help="run tests that require running adala server",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "use_openai: mark test as requiring OpenAI key")
    config.addinivalue_line(
        "markers", "use_server: mark test as requiring running adala server"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--use-openai"):
        skip_openai = pytest.mark.skip(reason="need OpenAI key to run")
        for item in items:
            if "use_openai" in item.keywords:
                item.add_marker(skip_openai)

    if not config.getoption("--use-server"):
        skip_server = pytest.mark.skip(reason="need live server to run")
        for item in items:
            if "use_server" in item.keywords:
                item.add_marker(skip_server)




@pytest.fixture
def client():
    os.environ['SINGLE_PRODUCER'] = 'true'
    from server.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def multiclient():
    os.environ['SINGLE_PRODUCER'] = 'false'

    from server.app import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def async_client():
    from server.app import app
    os.environ['SINGLE_PRODUCER'] = 'true'
    async with httpx.AsyncClient(
        timeout=10, app=app, base_url="http://localhost:30001"
    ) as client:
        yield client

@pytest_asyncio.fixture
async def multi_async_client():
    from server.app import app
    os.environ['SINGLE_PRODUCER'] = 'true'
    async with httpx.AsyncClient(
        timeout=10, app=app, base_url="http://localhost:30001"
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
