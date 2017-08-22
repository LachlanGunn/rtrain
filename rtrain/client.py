import requests
import requests_toolbelt.adapters.host_header_ssl
import sys
import time
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


class RTrainSession(object):
    def __init__(self, url, certificate=None, tls_host='rtraind'):
        self.url = url
        self.session = requests.Session()

        if certificate is not None:
            self.verify = certificate

        self.session.mount("https://", requests_toolbelt.adapters.host_header_ssl.HostHeaderSSLAdapter())
        self.host = tls_host

    def train(self, model, loss, optimizer, x_train, y_train, epochs, batch_size, quiet=False):
        """Train a model on a remote server."""
        global progressbar_type
        global notebook

        serialized_model = serialize_training_job(model, loss, optimizer, x_train, y_train, epochs, batch_size)
        response = self.session.post("%s/train" % self.url, json=serialized_model,
                                     verify=self.verify, headers={'Host': self.host})
        job_id = response.text

        if not quiet:
            if notebook:
                bar = progressbar_type(desc="Training Remotely", total=1000.0, unit='‰', mininterval=0)
            else:
                bar = progressbar_type(desc="Training Remotely", total=1000.0, unit='‰', mininterval=0,
                                       bar_format='{desc}{percentage:3.0f}% |{bar}| {elapsed} ({remaining} rem.)')

        finished = False
        last_status = 0
        while not finished:
            response = self.session.get("%s/status/%s" % (self.url, job_id),
                                        verify=self.verify, headers={'Host': self.host})
            if response.status_code != 200:
                print("Job not created.", file=sys.stderr)
                return None
            status = response.json()
            if status.get('error', None) is not None:
                raise IOError(status['error'])

            if not quiet:
                bar.update(int(round(10 * status['status'])) - last_status)
                last_status = int(round(10 * status['status']))

            finished = status['finished']
            time.sleep(5)

        if not quiet:
            bar.close()

        response = self.session.get("%s/result/%s" % (self.url, job_id),
                                    verify=self.verify, headers={'Host': self.host})
        return deserialize_model(response.text)
