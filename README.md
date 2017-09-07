The _rtrain_ Remote Trainer for Keras
=====================================

___This software is highly experimental.  Expect it to break at every
    opportunity.___

The **rtrain** package is a Python package for remotely training Keras
neural network models.  This allows scripts and notebooks to
remain conveniently located on the user's local machine, with just the
expensive training operation being offloaded to distant lands.

This approach has some advantages:

 - _Compared to using distributed TensorFlow._
    One is not tied to a particular backend; TensorFlow has some distributed
    capabilities, but this means that we cannot, for example, use Theano.
    
 - _Compared to a remote Jupyter Notebook._  The notebook itself stays
    safely on the local machine.  This ensures that it is
    properly backed up and does not require the use of a relatively
    expensive EC2 instance in order to run the parts of the notebook
    that do not require GPU acceleration, which may be the vast majority.
    
An example of the usage of the `rtrain` module is given in
[`remote_train_example.py`](remote_train_example.py).

Installing _rtrain_
-----------------

Install _rtrain_ using `pip` or `pip3` depending on your Python
environment:

```ShellSession
$ pip3 install git+https://github.com/LachlanGunn/rtrain
``` 

This will install the `rtrain` Python module, as well as the server-side
applications, which will be installed into `PATH`.

Using `rtraind`
---------------

You will need to run `rtraind` on the server responsible for the actual
computation.  For the moment it is necessary to manually initialise
its database.

***Always use an SSH tunnel or similar to communicate with `rtraind`.***

*A _Keras_ model can be used to execute arbitrary code, and thus
`rtraind` must not be exposed to outside users under any
circumstances.*

There are two main ways to do this:
 * *SSH tunnelling.*  This is easiest and handles authentication as
    well.
 * *TLS+password authentication.*  This requires a TLS-supporting
    reverse proxy.  Authentication is provided by the `Password`
    option in the configuration file.

In either case, until a sandboxing mechanism  is in place, it is safest
to assume that all authenticated users can see each others data if
they so desire.

### Setting up the server

We use _SQLAlchemy_ to store job data; the database location is stored in the
configuration file `/etc/rtraind.conf`, and does not need to exist at the time
of setup.  The database schema is initialised using `rtraind-setup`.

The configuration file looks as follows:

```ini
[rtraind]
Database=sqlite:///path/to/database.sqlite
Password=YouCanLeaveMeBlankToDisableAuthentication
```
and should be placed at `/etc/rtraind.conf`.  Then, we can run `rtraind-setup`,
```ShellSession
$ rtraind-setup
```
which will initialise the database.  If the config file is at another
location, use the `-c` or `--config` options:
```ShellSession
$ rtraind-setup -c /path/to/config
$ rtraind-setup --config /path/to/config
```

### Running the daemon

We are now ready to run the daemon:
```ShellSession
$ rtraind
```
As yet there is no `systemd` script, no logging, nor a proper
configuration file. The listen port cannot yet be changed, and defaults
to port 5000.  You should use a reverse proxy; this allows for TLS support
as well.

### Sending jobs

*A complete example is given in
[`remote_train_example.py`](remote_train_example.py).*

Normally one would train a model using `model.compile()` followed by
`model.train()`.  These are replaced with a single method,
`rtrain.client.RtrainSession.train()`:

```python
>>> session = rtrain.client.RTrainSession("http://localhost:5000")
>>> trained_model = session.train(model, 'mean_squared_error', 'rmsprop',
...                               x_train, y_train, 100, 128)       
``` 

This will return a trained version of the model; a progress bar will mark
the progress of its training.

Jupyter notebook support can be enabled with `rtrain.set_notebook(True)`.
This results in a more attractive progress bar.

The Author
----------

**Lachlan Gunn**

<table>
<tr><td>Email</td><td>lachlan@twopif.net</td></tr>
<tr>
    <td>PGP</td>
    <td><code>F3E3 8891 8560 5B82 933D  6180 D288 91D2 136B 33B0</code></td>
</tr>
<tr>
    <td>Github</td>
    <td><a href="https://github.com/lachlangunn">LachlanGunn</a></td>
</tr>
<tr>
    <td>ORCID</td>
    <td><a href="https://orcid.org/0000-0003-1767-7897">0000-0003-1767-7897</a></td>
</tr>
</table>
