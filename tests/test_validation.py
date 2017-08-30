#!/usr/bin/env python

import rtrain.validation


def test_validation():
    # Not enough properties.
    assert not rtrain.validation.validate_training_request({})
    assert not rtrain.validation.validate_training_request({
        "architecture":
        "",
        "weights": ["yay_for_arrays"]
    })

    # Not only not enough properties, but those that are there are wrong.
    assert not rtrain.validation.validate_training_request({
        "architecture": "",
        "weights": [0]
    })

    # Perfect!
    assert rtrain.validation.validate_training_request({
        "architecture":
        "",
        "weights": ["yay_for_arrays"],
        "loss":
        "mean_squared_error",
        "optimizer":
        "rmsprop",
        "x_train":
        "more array",
        "y_train":
        "more array",
        "x_train_shape": [3],
        "y_train_shape": [3],
        "epochs":
        10,
        "batch_size":
        1
    })

    # Too many properties.
    assert not rtrain.validation.validate_training_request(
        {
            "architecture": "",
            "weights": ["yay_for_arrays"],
            "loss": "mean_squared_error",
            "optimizer": "rmsprop",
            "x_train": "more array",
            "y_train": "more array",
            "x_train_shape": [3],
            "y_train_shape": [3],
            "epochs": 10,
            "batch_size": 1,
            "ham": False
        })
