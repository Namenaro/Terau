# -*- coding: utf-8 -*
import theano
import theano.tensor
floatX = theano.config.floatX

import lasagne
import lasagne.layers
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(42)

def draw_trajectory(dropout_regressor, from_, to_, steps):
    X = np.linspace(from_, to_, steps)
    X = np.array(X).astype(floatX)
    X = np.matrix(X).T
    x = theano.tensor.matrix('x', dtype=theano.config.floatX)
    prediction = lasagne.layers.get_output(dropout_regressor.model, x)
    f = theano.function([x], prediction)
    output = f(X)
    X = np.asarray(X).reshape(-1)
    output = output.flatten()
    plt.scatter(X, output, c='k', label='data', zorder=1)


def draw_process(dropout_regressor, from_, to_, steps):
    x = from_
    dx = (to_ - from_) / float(steps)
    for i in range(steps):
        print "next x point.."
        # отрисовываем вертикальные палочки
        mu, std = dropout_regressor.predict(x)
        half_std = float(std / 2)
        print mu
        print std
        print x
        plt.plot([x, x], [mu - half_std, mu + half_std], 'k-', lw=2)
        x += dx