# -*- coding: utf-8 -*
import theano
import theano.tensor
floatX = theano.config.floatX

import lasagne
import lasagne.layers
import matplotlib.pyplot as plt

import numpy as np
import dropout_regressor
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

def visualize_dropout_regressor(model_file, info_file, trajectories=4, from_=-1.5, to_=1.5, steps=15):
    P, X, Y = dropout_regressor.DropoutRegressor.get_info_from_file(info_file)
    model = dropout_regressor.DropoutRegressor(file_with_model=model_file)
    plt.figure()
    for i in range(trajectories):
        draw_trajectory(model, from_=from_, to_=to_, steps=steps)
    plt.scatter(X, Y, c='r', label='real_data', zorder=1)
    plt.savefig(model_file + "_visualisation.png")