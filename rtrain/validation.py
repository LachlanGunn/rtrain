#!/usr/bin/env python3
"""Request validation."""

import jsonschema

schema = {
    "$id":
    "http://twopif.net/rtrain/schema/training-job/1.0",
    "definitions": {
        "ndarray": {
            "type": "string"
        }
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
    """Validate a JSON-formatted training request."""
    return jsonschema.Draft4Validator(schema).is_valid(request)
