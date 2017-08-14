import time

import requests
import tqdm

from rtrain.utils import serialize_training_job, deserialize_model

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


def train(url, model, loss, optimizer, x_train, y_train, epochs, batch_size, quiet=False):
    """Train a model on a remote server."""
    global progressbar_type
    global notebook

    serialized_model = serialize_training_job(model, loss, optimizer, x_train, y_train, epochs, batch_size)
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

    response = requests.get("%s/result/%s" % (url, job_id))
    return deserialize_model(response.text)