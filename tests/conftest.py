import os
import pytest
from unittest import mock
from fakeredis import FakeStrictRedis


@pytest.fixture
def redis_mock(monkeypatch):
    fake_redis = FakeStrictRedis()
    monkeypatch.setattr("server.app.Redis", mock.Mock())
    monkeypatch.setattr("server.app.Redis.from_url", lambda *args, **kwargs: fake_redis)
    return fake_redis


@pytest.fixture
def celery_app_mock(monkeypatch, celery_app):
    monkeypatch.setattr("server.tasks.process_file.app", celery_app)
    return celery_app


@pytest.fixture
def openai_key_mock():
    key = "mocked"
    os.environ["OPENAI_API_KEY"] = key
    return key
