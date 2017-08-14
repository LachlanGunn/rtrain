import base64
import io
import json

import keras.models
import numpy


def serialize_array(array):
    f = io.BytesIO()
    numpy.save(f, array)
    return str(base64.b64encode(f.getvalue()), 'ascii')


def deserialize_array(s):
    f = io.BytesIO(base64.b64decode(s))
    return numpy.load(f)


def serialize_model(model):
    """Serialize a Keras model into JSON."""
    architecture = model.to_json()
    weights = model.get_weights()

    # We need to convert the weights to JSON
    weights_lists = [serialize_array(x) for x in weights]

    return json.dumps({'architecture': architecture, 'weights': weights_lists})


def deserialize_model(model_json):
    """Deserialize a Keras model from JSON."""
    parsed_model = json.loads(model_json)
    model = keras.models.model_from_json(parsed_model['architecture'])
    model.set_weights([deserialize_array(w) for w in parsed_model['weights']])
    return model


def serialize_training_job(model, loss, optimizer, x_train, y_train, epochs, batch_size):
    architecture = model.to_json()
    weights = model.get_weights()

    # We need to convert the arrays to strings
    weights_serialized = [serialize_array(w) for w in weights]

    return ({'architecture': architecture, 'weights': weights_serialized,
             'loss': loss, 'optimizer': optimizer,
             'x_train': serialize_array(x_train), 'y_train': serialize_array(y_train),
             'x_train_shape': x_train.shape, 'y_train_shape': y_train.shape,
             'epochs': epochs, 'batch_size': batch_size})