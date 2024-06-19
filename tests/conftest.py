import pytest
from unittest import mock
from fakeredis import FakeStrictRedis


@pytest.fixture
def redis_mock(monkeypatch):
    fake_redis = FakeStrictRedis()
    monkeypatch.setattr("server.app.Redis", lambda *args, **kwargs: fake_redis)
    return fake_redis
