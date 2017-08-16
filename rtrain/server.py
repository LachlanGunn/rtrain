#!/usr/bin/env python3

import flask
import json
import keras.models
import sqlite3
import sys
import threading
import time
import traceback

from rtrain.utils import deserialize_array, serialize_model
from .validation import validate_training_request
import rtrain.server_utils.database_operations as _database_operations

app = flask.Flask(__name__)

database_path = _database_operations.get_database_location()


def get_db():
    """Get a database context."""
    if not hasattr(flask.g, 'sqlite_db'):
        flask.g.sqlite_db = sqlite3.connect(database_path)
    return flask.g.sqlite_db


@app.teardown_appcontext
def close_db(exception):
    """Close the database when the application closes."""
    if hasattr(flask.g, 'sqlite_db'):
        flask.g.sqlite_db.close()


def extract_training_request(json_data):
    if not validate_training_request(json_data):
        return None
    else:
        return json_data


def execute_training_request(training_job, callback):
    model = keras.models.model_from_json(training_job['architecture'])
    model.compile(loss=training_job['loss'], optimizer=training_job['optimizer'])

    model.set_weights([deserialize_array(w) for w in training_job['weights']])
    x_train = deserialize_array(training_job['x_train'])
    y_train = deserialize_array(training_job['y_train'])

    model.fit(x_train, y_train, epochs=training_job['epochs'], callbacks=[callback], verbose=0,
              batch_size=training_job['batch_size'])
    return serialize_model(model)


class StatusCallback(keras.callbacks.Callback):
    def __init__(self, job_id, db):
        self.db = db
        self.job_id = job_id
        self.epochs_finished = 0
        self.samples_this_epoch = 0
        self.last_update = -1

    def on_epoch_begin(self, epoch, logs={}):
        self.samples_this_epoch = 0

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.samples_this_epoch += batch_size

        current_time = time.time()
        if current_time - self.last_update > 0.5:
            _database_operations.update_status(self.job_id, 100.0 * (float(self.samples_this_epoch)/self.params['samples'] + self.epochs_finished) / self.params['epochs'], self.db)
            self.last_update = current_time

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_finished += 1


def trainer():
    database = sqlite3.connect(database_path)
    while True:
        next_job = _database_operations.get_next_job(database)
        if next_job is None:
            time.sleep(1)
            continue

        job_id, job = next_job
        print("Starting job %s" % job_id, file=sys.stderr)
        training_request = extract_training_request(json.loads(job))
        callback = StatusCallback(job_id, database)
        try:
            result = execute_training_request(training_request, callback)
            _database_operations.finish_job(job_id, result, database)
        except Exception as e:
            _database_operations.update_status(job_id, -1, database)
            _database_operations.finish_job(job_id, traceback.format_exc(e), database)


def cleaner():
    database = sqlite3.connect(database_path)
    while True:
        _database_operations.purge_old_jobs(database)
        time.sleep(30)


@app.route("/train", methods=['POST'])
def request_training():
    request_content = flask.request.get_json()
    if request_content is None:
        flask.abort(415)
    training_request = extract_training_request(request_content)
    return _database_operations.create_new_job(training_request, get_db())


@app.route("/status/<job_id>", methods=['GET'])
def request_status(job_id):
    status = _database_operations.get_status(job_id, get_db())
    if status is None:
        flask.abort(404)
    else:
        return json.dumps({'status': status[0], 'finished': status[1]})


@app.route("/result/<job_id>", methods=['GET'])
def request_result(job_id):
    status = _database_operations.get_results(job_id, get_db())
    if status is None:
        flask.abort(404)
    else:
        return status[0]


def main():
    worker_thread = threading.Thread(target=trainer)
    worker_thread.start()

    cleaner_thread = threading.Thread(target=cleaner)
    cleaner_thread.start()

    app.run()


if __name__ == '__main__':
    main()
