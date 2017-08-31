#!/usr/bin/env python3
"""Keras remote-training server."""

import argparse
from functools import wraps
import json
import logging
import os
import sys
import threading
import time
import traceback

import flask
import keras.models
import sqlalchemy.orm

import structlog
import structlog.stdlib

import rtrain.server_utils.config
import rtrain.server_utils.model
import rtrain.server_utils.model.database_operations as _database_operations

from rtrain.utils import deserialize_array, serialize_model
from rtrain.validation import validate_training_request

rtraind_blueprint = flask.Blueprint('rtraind', __name__)

Session = None
password = None

logger = structlog.get_logger()

# Tell TensorFlow to be quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_app(config):
    app = flask.Flask(__name__)
    app.register_blueprint(rtraind_blueprint)
    return app


def prepare_database(config):
    """Prepare a database connection given a database string."""
    global Session
    engine = sqlalchemy.create_engine(config.db_string)
    session_factory = sqlalchemy.orm.sessionmaker(bind=engine)
    Session = sqlalchemy.orm.scoped_session(session_factory)


def extract_training_request(json_data):
    """Validate a training request."""
    if not validate_training_request(json_data):
        return None

    return json_data


def execute_training_request(training_job, callback):
    """Execute a deserialised training request, returning a trained model."""
    model = keras.models.model_from_json(training_job['architecture'])
    model.compile(
        loss=training_job['loss'], optimizer=training_job['optimizer'])

    model.set_weights([deserialize_array(w) for w in training_job['weights']])
    x_train = deserialize_array(training_job['x_train'])
    y_train = deserialize_array(training_job['y_train'])

    model.fit(
        x_train,
        y_train,
        epochs=training_job['epochs'],
        callbacks=[callback],
        verbose=0,
        batch_size=training_job['batch_size'])
    return serialize_model(model)


##########################################################################
# Based on http://flask.pocoo.org/snippets/8/
####
# "This snippet by Armin Ronacher can be used freely for anything you like.
#  Consider it public domain."
##########################################################################
def check_auth(_, http_password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return (password is not None) and (password == http_password)


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return flask.Response('Login required.', 401,
                          {'WWW-Authenticate': 'Basic realm="Login Required"'})


def requires_auth(f):
    """A decorator that forces authentication when a password is set."""

    @wraps(f)
    def decorated(*args, **kwargs):
        global password
        if password is None:
            return f(*args, **kwargs)
        auth = flask.request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


##########################################################################


class StatusCallback(keras.callbacks.Callback):
    """A callback class to store job status to the database."""

    def __init__(self, job_id, db):
        self.db = db
        self.job_id = job_id
        self.epochs_finished = 0
        self.samples_this_epoch = 0
        self.last_update = -1

    def on_epoch_begin(self, epoch, logs=None):
        self.samples_this_epoch = 0

    def on_batch_end(self, batch, logs=None):
        batch_size = logs.get('size', 0)
        self.samples_this_epoch += batch_size

        current_time = time.time()
        if current_time - self.last_update > 0.5:
            _database_operations.update_status(
                self.job_id, 100.0 *
                (float(self.samples_this_epoch) / self.params['samples'] +
                 self.epochs_finished) / self.params['epochs'], self.db)
            self.last_update = current_time

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_finished += 1


def trainer():
    """Thread that performs the actual model training."""
    session = Session()
    log = logger.new()
    while True:
        log.debug('trainer::job::wait_for_next')
        while True:
            next_job = _database_operations.get_next_job(session)
            if next_job is None:
                time.sleep(1)
                continue
            else:
                break

        job_log = log.bind(job_id=next_job.id)

        job = next_job
        if not job.training_jobs:
            job_log.warn('trainer::job::no_training_job')
            continue

        job_log.info('trainer::job::job_start')
        for i, tj in enumerate(job.training_jobs):
            subjob_log = job_log.bind(subjob_type='training', subjob=i)
            subjob_log.info('trainer::job::subjob_start')
            training_data = str(tj.training_job, 'utf8')
            training_request = extract_training_request(
                json.loads(training_data))
            callback = StatusCallback(job.id, session)
            try:
                result = execute_training_request(training_request, callback)
                _database_operations.finish_job(job.id, result, session)
            except Exception as e:
                subjob_log.error('trainer::job::error', exc_info=True)
                _database_operations.update_status(job.id, -1, session)
                _database_operations.finish_job(job.id,
                                                traceback.format_exc(e),
                                                session)
            subjob_log.info('trainer::job::subjob_finished')
        job_log.info('trainer::job::job_finished')


def cleaner():
    """Thread that purges old jobs from the database."""
    session = Session()
    while True:
        _database_operations.purge_old_jobs(session)
        time.sleep(30)


@rtraind_blueprint.route("/ping")
def ping():
    """Basic health check request."""
    return '{}'


@rtraind_blueprint.route("/train", methods=['POST'])
@requires_auth
def request_training():
    """Request handler for training requests."""
    log = logger.new()
    request_content = flask.request.get_json()
    if request_content is None:
        log.error('frontend::train_request::invalid_json')
        flask.abort(415)

    training_request = extract_training_request(request_content)
    if training_request is None:
        log.error('frontend::train_request::invalid_request')
        flask.abort(400)

    job_id = _database_operations.create_new_job(training_request, Session())
    log.info('frontend::train_request::request_training', job_id=job_id)
    return job_id


@rtraind_blueprint.route("/status/<job_id>", methods=['GET'])
@requires_auth
def request_status(job_id):
    """Handler for job status requests."""
    status = _database_operations.get_status(job_id, Session())
    if status is None:
        flask.abort(404)
    else:
        return json.dumps({
            'status': status.status,
            'finished': status.finished
        })


@rtraind_blueprint.route("/result/<job_id>", methods=['GET'])
@requires_auth
def request_result(job_id):
    """Handler for job result downloads."""
    result = _database_operations.get_results(job_id, Session())
    if result is None:
        flask.abort(404)
    else:
        return result


def main():
    global password

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        default='/etc/rtraind.conf',
        help='Path to configuration file.')
    args = parser.parse_args()

    log = logger.new()

    try:
        with open(args.config) as config_fh:
            config = rtrain.server_utils.config.RTrainConfig(config_fh.read())
    except FileNotFoundError:
        log.warn(
            'startup::config_file::file_not_found', config_file=args.config)
        config = rtrain.server_utils.config.RTrainConfig('')
    except IOError:
        log.fatal("startup::config_file::read_failed", config_file=args.config)
        sys.exit(1)

    prepare_database(config)
    password = config.password

    worker_thread = threading.Thread(target=trainer)
    worker_thread.start()

    cleaner_thread = threading.Thread(target=cleaner)
    cleaner_thread.start()

    app = create_app(config)
    app.run()


if __name__ == '__main__':
    main()
