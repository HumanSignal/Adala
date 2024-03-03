import pytest
from fastapi.testclient import TestClient
from adala.server.app import app
from adala.agents import Agent
from adala.server.tasks.process_file import process_file
from unittest.mock import patch

client = TestClient(app)

def test_submit_endpoint_with_valid_request():
    with patch.object(process_file, 'delay') as mock_task:
        mock_task.return_value.id = '123'
        response = client.post("/submit", json={"agent": Agent(), "task_name": "process_file"})
        assert response.status_code == 200
        assert response.json() == {"success": True, "data": {"job_id": "123"}}

def test_submit_endpoint_with_invalid_request():
    response = client.post("/submit", json={"agent": Agent(), "task_name": "invalid_task"})
    assert response.status_code == 422

def test_get_status_endpoint_with_valid_request():
    with patch.object(process_file, 'AsyncResult') as mock_task:
        mock_task.return_value.status = 'PENDING'
        response = client.post("/get-status", json={"job_id": "123"})
        assert response.status_code == 200
        assert response.json() == {"success": True, "data": {"status": "PENDING"}}

def test_get_status_endpoint_with_invalid_request():
    response = client.post("/get-status", json={"job_id": ""})
    assert response.status_code == 422

def test_cancel_job_endpoint_with_valid_request():
    with patch.object(process_file, 'AsyncResult') as mock_task:
        mock_task.return_value.revoke.return_value = None
        response = client.post("/cancel", json={"job_id": "123"})
        assert response.status_code == 200
        assert response.json() == {"success": True, "data": {"status": "cancelled"}}

def test_cancel_job_endpoint_with_invalid_request():
    response = client.post("/cancel", json={"job_id": ""})
    assert response.status_code == 422