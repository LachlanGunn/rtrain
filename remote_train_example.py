#!/usr/bin/env python3

import numpy
import rtrain

from keras.models import Sequential
from keras.layers import Dense

if __name__ == '__main__':
    model = Sequential([
        Dense(32, activation='tanh', input_shape=(2,)),
        Dense(1),
    ])

    x_train = numpy.random.randn(10000,2)
    y_train = numpy.matrix(numpy.sqrt(x_train[:,0]**2 + x_train[:,1]**2)).transpose()
    print(y_train.shape)

    trained_model = rtrain.train("http://localhost:5000", model, 'mean_squared_error', 'rmsprop', x_train, y_train,
                                 100, 128)

    print(numpy.sqrt(0.3**2 + 0.6**2))
    print(trained_model.predict(numpy.matrix([[0.3, 0.6]])))