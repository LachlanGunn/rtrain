#!/usr/bin/env python3
"""Request validation."""

import jsonschema

schema = {
    "$id":
    "http://twopif.net/rtrain/schema/training-job/1.0",
    "definitions": {
        "ndarray": {
            "type": "string",
        },
    },
    "type":
    "object",
    "required": [
        "architecture", "weights", "loss", "optimizer", "epochs", "x_train",
        "y_train", "x_train_shape", "y_train_shape"
    ],
    "additionalProperties":
    False,
    "properties": {
        "architecture": {
            "type": "string"
        },
        "weights": {
            "type": "array",
            "items": {
                "$ref": "#/definitions/ndarray"
            }
        },
        "loss": {
            "type": "string"
        },
        "optimizer": {
            "type": "string"
        },
        "x_train": {
            "$ref": "#/definitions/ndarray"
        },
        "y_train": {
            "$ref": "#/definitions/ndarray"
        },
        "x_train_shape": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1
            }
        },
        "y_train_shape": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1
            }
        },
        "epochs": {
            "type": "integer"
        },
        "batch_size": {
            "type": "integer",
            "minimum": 1
        }
    }
}


def validate_training_request(request):
    """Validate a JSON-formatted training request.

    # Not enough properties.
    >>> validate_training_request({})
    False

    # Still not enough properties.
    >>> validate_training_request({"architecture":"","weights":[[[1,2,3]],[[4,5,6]]]})
    False

    # Not only not enough properties, but those that are there are wrong.
    >>> validate_training_request({"architecture":"","weights":["ham"]})
    False

    # Perfect!
    >>> validate_training_request({"architecture":"","weights":[[[1,2,3]],[[4,5,6]]],
    ...                            "loss": "mean_squared_error", "optimizer": "rmsprop",
    ...                             "x_train": [1,2,3], "y_train": [4,5,6],
    ...                             "x_train_shape": [3], "y_train_shape": [3],
    ...                             "epochs": 10, "batch_size": 1})
    True

    # Too many properties.
    >>> validate_training_request({"architecture":"","weights":[[[1,2,3]],[[4,5,6]]],
    ...                            "loss": "mean_squared_error", "optimizer": "rmsprop",
    ...                             "x_train": [1,2,3], "y_train": [4,5,6],
    ...                             "x_train_shape": [3], "y_train_shape": [3],
    ...                             "epochs": 10, "batch_size": 1,
    ...                             "ham": False})
    False
    """
    return jsonschema.Draft4Validator(schema).is_valid(request)
