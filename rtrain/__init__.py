#!/usr/bin/env python3

"""Train Keras models remotely."""

import base64
import io
import json
import keras.models
import numpy
import requests
import time
import tqdm

progressbar_type = tqdm.tqdm
notebook = False


def set_notebook(in_notebook):
    """Specify whether or not rtrain should use Jupyter Notebook widgets."""
    global progressbar_type
    global notebook
    notebook = in_notebook
    if in_notebook:
        progressbar_type = tqdm.tqdm_notebook
    else:
        progressbar_type = tqdm.tqdm


def _serialize_array(array):
    f = io.BytesIO()
    numpy.save(f, array)
    return str(base64.b64encode(f.getvalue()), 'ascii')


def _deserialize_array(s):
    f = io.BytesIO(base64.b64decode(s))
    return numpy.load(f)


def serialize_model(model):
    """Serialize a Keras model into JSON."""
    architecture = model.to_json()
    weights = model.get_weights()

    # We need to convert the weights to JSON
    weights_lists = [_serialize_array(x) for x in weights]

    return json.dumps({'architecture': architecture, 'weights': weights_lists})


def deserialize_model(model_json):
    """Deserialize a Keras model from JSON."""
    parsed_model = json.loads(model_json)
    model = keras.models.model_from_json(parsed_model['architecture'])
    model.set_weights([_deserialize_array(w) for w in parsed_model['weights']])
    return model


def _serialize_training_job(model, loss, optimizer, x_train, y_train, epochs, batch_size):
    architecture = model.to_json()
    weights = model.get_weights()

    # We need to convert the arrays to strings
    weights_serialized = [_serialize_array(w) for w in weights]

    return ({'architecture': architecture, 'weights': weights_serialized,
             'loss': loss, 'optimizer': optimizer,
             'x_train': _serialize_array(x_train), 'y_train': _serialize_array(y_train),
             'x_train_shape': x_train.shape, 'y_train_shape': y_train.shape,
             'epochs': epochs, 'batch_size': batch_size})


def train(url, model, loss, optimizer, x_train, y_train, epochs, batch_size, quiet=False):
    """Train a model on a remote server."""
    global progressbar_type
    global notebook

    serialized_model = _serialize_training_job(model, loss, optimizer, x_train, y_train, epochs, batch_size)
    response = requests.post("%s/train" % url, json=serialized_model)
    job_id = response.text

    if not quiet:
        if notebook:
            bar = progressbar_type(desc="Training Remotely", total=100.0, unit='%', mininterval=0)
        else:
            bar = progressbar_type(desc="Training Remotely", total=100.0, unit='%', mininterval=0,
                                   bar_format='{desc}{percentage:3.0f}% |{bar}| {elapsed} ({remaining} rem.)')

    finished = False
    last_status = 0
    while not finished:
        response = requests.get("%s/status/%s" % (url, job_id))
        status = response.json()
        if status.get('error', None) is not None:
            raise IOError(status['error'])

        if not quiet:
            bar.update(int(round(status['status'])) - last_status)
            last_status = int(round(status['status']))

        finished = status['finished']
        time.sleep(5)

    if not quiet:
        bar.close()

    response = requests.get("%s/result/%s" % (url,job_id))
    return deserialize_model(response.text)