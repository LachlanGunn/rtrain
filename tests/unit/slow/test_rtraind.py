#!/usr/bin/env python3

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


def test_status_fail_badjob(client, monkeypatch):
    monkeypatch.setattr('rtrain.server.Session', lambda: None)
    monkeypatch.setattr(
        'rtrain.server_utils.model.database_operations.get_status',
        lambda x, y: None)
    result = client.get(
        flask.url_for('rtraind.request_status', job_id='not_a_real_id'))
    assert result.status_code == 404


def test_status_success(client, monkeypatch):
    class Status(object):
        """Class to replace the SQLAlchemy model object."""

        def __init__(self, status, finished):
            self.status = status
            self.finished = finished

        def test_func(self, test_job_id):
            """Return a stub function for get_status that checks job_id."""

            def f(job_id, _):
                assert job_id == test_job_id
                return self

            return f

    # Stub out the database.
    monkeypatch.setattr('rtrain.server.Session', lambda: None)

    # Test with one value.
    monkeypatch.setattr(
        'rtrain.server_utils.model.database_operations.get_status',
        Status(3.14159, False).test_func('a_real_id'))
    result = client.get(
        flask.url_for('rtraind.request_status', job_id='a_real_id'))
    assert result.status_code == 200
    assert result.json['status'] == 3.14159
    assert not result.json['finished']

    # Test with another value to make sure it isn't just a constant function.
    monkeypatch.setattr(
        'rtrain.server_utils.model.database_operations.get_status',
        Status(2.71, True).test_func('another_real_id'))
    result = client.get(
        flask.url_for('rtraind.request_status', job_id='another_real_id'))
    assert result.status_code == 200
    assert result.json['status'] == 2.71
    assert result.json['finished']


def test_results_badjob(client, monkeypatch):
    # Stub out the database.
    monkeypatch.setattr('rtrain.server.Session', lambda: None)

    def get_check_job_id(desired_job_id):
        def check_job_id(job_id, _):
            assert job_id == desired_job_id
            return None

        return check_job_id

    # Mock the database call to return a failure.
    monkeypatch.setattr(
        'rtrain.server_utils.model.database_operations.get_results',
        get_check_job_id('not_a_real_id'), )

    result = client.get(
        flask.url_for('rtraind.request_result', job_id='not_a_real_id'))

    assert result.status_code == 404


def test_results_success(client, monkeypatch):
    # Stub out the database.
    monkeypatch.setattr('rtrain.server.Session', lambda: None)

    def perform_test(job_id, result):
        """Test /result/XXX in a way that should succeed."""

        def get_check_job_id(desired_job_id):
            def check_job_id(internal_job_id, _):
                assert internal_job_id == desired_job_id
                return result

            return check_job_id

        # Mock the database call to return the desired value.
        monkeypatch.setattr(
            'rtrain.server_utils.model.database_operations.get_results',
            get_check_job_id(job_id))

        # Run the function under test.
        response = client.get(
            flask.url_for('rtraind.request_result', job_id=job_id))

        # Check the response.
        assert response.status_code == 200
        assert str(response.data, 'utf8') == result

    # Now perform the actual test with several input values to make
    # sure it is really doing something.
    perform_test('the_first_real_id', "A result")
    perform_test('the_second_real_id', "Another result")
