import pytest
from unittest import mock
from fakeredis import FakeStrictRedis


@pytest.fixture
def redis_mock(monkeypatch):
    fake_redis = FakeStrictRedis()
    monkeypatch.setattr("server.app.Redis", mock.Mock())
    monkeypatch.setattr("server.app.Redis.from_url", lambda *args, **kwargs: fake_redis)
    return fake_redis
