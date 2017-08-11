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
    expensive AWS instance in order to run the parts of the notebook
    that do not require acceleration, which may be the vast majority.
    
An example of the usage of the `rtrain` module is given in
[`remote_train_example.py`](remote_train_example.py).

Building _rtrain_
-----------------

Build and install the Python wheel file with

```ShellSession
rtrain$ python3 setup.py bdist_wheel
rtrain$ sudo pip3 install dist/rtrain-0.0.1-py3-none-any.whl
``` 

This will install the `rtrain` python module, as well as the server-side
applications.

Using `rtraind`
---------------

You will need to run `rtraind` on the server responsible for the actual
computation.  For the moment it is necessary to manually initialise
its database.

***Always use an SSH tunnel or similar to communicate with `rtraind`.***

*There is currently no
authentication or encryption, and I do not know for certain that a
_Keras_ model cannot be used to execute arbitrary code&mdash;features
like user-defined layer types make this likely&mdash;and thus `rtraind`
must not be exposed to outside users under any circumstances.*

### Setting up the server

We use _SQLite_ to store job data; the database location is stored in the
environmental variable `DB_PATH`, and does not need to exist at the time
of setup.  The database schema is initialised using `rtraind-setup`.  

```ShellSession
$ DB_PATH="/path/to/database.sqlite" rtraind-setup
```

### Running the daemon

We are now ready to run the daemon:
```ShellSession
$ DB_PATH="/path/to/database.sqlite" rtraind
```
As yet there is no `systemd` script, no logging, nor a proper
configuration file. The listen port cannot yet be changed, however
by default `rtraind` will listen on port 5000.

### Sending jobs

*A complete example is given in
[`remote_train_example.py`](remote_train_example.py).*

Normally one would train a model using `model.compile()` followed by
`model.train()`.  These are replaced with a single function, `rtrain.train()`:

```python
>>> trained_model = rtrain.train("http://localhost:5000", model,
...     'mean_squared_error', 'rmsprop', x_train, y_train, 100, 128)       
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
