#!/usr/bin/env python

import flask
import pytest

import rtrain.server
import rtrain.server_utils.model.database_operations
import rtrain.utils


@pytest.fixture
def app():
    return rtrain.server.create_app(None)


def test_ping(client):
    result = client.get(flask.url_for('rtraind.ping'))
    assert result.status_code == 200
    assert result.json == {}


def test_train_badjson_fail(client, monkeypatch):
    result = client.post(
        flask.url_for('rtraind.request_training'),
        data='Hello',
        content_type='text/plain')
    assert result.status_code == 415


def test_train_badrequest_success(client, monkeypatch):
    monkeypatch.setattr('rtrain.server.extract_training_request',
                        lambda x: None)
    result = client.post(
        flask.url_for('rtraind.request_training'),
        data='{}',
        content_type='application/json')
    assert result.status_code == 400


def test_train_success(client, monkeypatch):
    def add_job(job_data, _):
        assert job_data == {}
        return '01234567890123456789012345678901'

    monkeypatch.setattr('rtrain.server.Session', lambda: None)
    monkeypatch.setattr('rtrain.server.extract_training_request', lambda x: {})
    monkeypatch.setattr(
        'rtrain.server_utils.model.database_operations.create_new_job',
        add_job)
    result = client.post(
        flask.url_for('rtraind.request_training'),
        data='{}',
        content_type='application/json')
    assert result.status_code == 200
    assert result.data == add_job({}, None).encode('utf8')
