import os
import pytest
from unittest import mock
from fakeredis import FakeStrictRedis, FakeRedis#, FakeConnectionPool
import openai_responses
from openai_responses import OpenAIMock


@pytest.fixture
def redis_mock(monkeypatch):
    fake_strict_redis = FakeStrictRedis()
    fake_redis = FakeRedis()
    # monkeypatch.setattr("server.app.Redis", mock.Mock())
    # monkeypatch.setattr("server.app.Redis.from_url", lambda *args, **kwargs: fake_redis)
    # # time to do something atrocious :D
    # # monkeypatch.setattr("server.tasks.process_file.app.", lambda *args, **kwargs: fake_redis)
    # return fake_redis
    import redis
    with (
        mock.patch.object(redis, 'StrictRedis', new=lambda *args, **kwargs: fake_strict_redis) as mocked_strict_redis,
        mock.patch.object(redis, 'Redis', new=lambda *args, **kwargs: fake_redis) as mocked_redis,
    ):
        yield mocked_strict_redis, mocked_redis

@pytest.fixture(scope='session')
def celery_config():
    return {
        'broker_url': 'memory://',
        'result_backend': 'redis://',
        'accept_content': ['json', 'pickle'],
    }


# @pytest.fixture(scope='session')
# def celery_worker_parameters():
    # return {
    # }


@pytest.fixture
def celery_app_mock(monkeypatch, redis_mock, celery_session_app, celery_worker, celery_config):
    monkeypatch.setattr("server.tasks.process_file.app", celery_session_app)
    # breakpoint()
    return celery_session_app


@pytest.fixture
def openai_key_mock():
    key = "mocked"
    os.environ["OPENAI_API_KEY"] = key
    return key


@pytest.fixture
def openai_mock_magic(monkeypatch):
    # monkeypatch.setattr("server.utils.openai", openai_mock)
    key = "mocked"
    # os.environ["OPENAI_API_KEY"] = key
    monkeypatch.setenv("OPENAI_API_KEY", key)

    openai_mock = OpenAIMock()
    openai_mock.router.route(host="localhost").pass_through()
    openai_mock.chat.completions.create.response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": "mocked openai chat response",
                    "role": "assistant",
                },
            }
        ]
    }
    # return openai_mock
    with openai_mock.router:
        yield openai_mock
