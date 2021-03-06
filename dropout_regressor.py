# -*- coding: utf-8 -*
import cPickle as pickle

import theano
import theano.tensor
import data_generator

floatX = theano.config.floatX

import lasagne
import lasagne.layers
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import get_output
from lasagne.objectives import squared_error

import numpy as np

import base_regressor
np.random.seed(43)

class DropoutRegressor (base_regressor.Regressor):
    def __init__(self, file_with_model=None):
        self.params = {}
        self.params['weight_decay'] = 0.00001
        self.params['learning_rate'] = 0.02
        self.params['num_iterations'] = 100
        self.params['dropout'] = 0.2
        self.params['num_neurons2'] = 20
        self.params['num_neurons3'] = 20
        self.params['predicive_sample_size'] = 10
        self.input_var = theano.tensor.matrix('input_var')
        self.target_var = theano.tensor.vector('target_var')
        if file_with_model is None:
            self.model = self.symbolic_droput_model()
        else:
            self.restore_model_from_file(file_with_model)
        self.train_function = self.symbolic_train_fn()
        self.learnedX = []
        self.learnedY = []

    def get_N(self):
        return len(self.learnedX)

    def symbolic_droput_model(self):
        input_layer = InputLayer(shape=(None, 1),
                                 name='input_layer',
                                 input_var=self.input_var)

        d2 = DenseLayer(incoming=input_layer,
                                  num_units=self.params['num_neurons2'],
                                  nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
                                  name='second_layer')

        dr2 = lasagne.layers.DropoutLayer(d2, p=self.params['dropout'])

        d3 = DenseLayer(incoming=dr2,
                        num_units=self.params['num_neurons3'],
                        nonlinearity=lasagne.nonlinearities.leaky_rectify,
                        name='second_layer')

        dr3 = lasagne.layers.DropoutLayer(d3, p=self.params['dropout'])

        output_layer = DenseLayer(incoming=dr3,
                                  num_units=1,
                                  nonlinearity=lasagne.nonlinearities.identity,
                                  name='output_layer')

        return output_layer

    def symbolic_train_fn(self):
        # символьная оптимизируемая функция
        predictions = get_output(self.model)
        loss = squared_error(predictions, self.target_var)  # возвращает 1d тензор
        loss = loss.mean()  # а вот теперь скаляр
        weights_L2 = lasagne.regularization.regularize_network_params(self.model, lasagne.regularization.l2)
        loss += self.params['weight_decay'] * weights_L2

        # какие параметры оптимизируем и как
        params = lasagne.layers.get_all_params(self.model, trainable=True)
        updates = lasagne.updates.adam(
            loss, params,
            learning_rate=self.params['learning_rate'])
        train_fn = theano.function(inputs=[self.input_var, self.target_var],
                                   outputs=loss,
                                   updates=updates,
                                   allow_input_downcast=True)  # float64 ->float32
        return train_fn

    def predict(self, x):
        """
        Скормить точку в модель и получить из нее предсказание и дисперсию
        :param point: точка данных
        :return: predictive_mean, predictive_var
        """
        samples = []
        for _ in range(self.params['predicive_sample_size']):
            ans = self.make_pred_in_one_point(x)
            samples.append(ans)
        return np.mean(samples), np.std(samples)

    def learn(self, x, y, learning_intensity):
        """
        Провести итерацию обучения регрессора по новой точке/ах (x, y) данных
        :param x:
        :param y:
        :param learning_intensity:
        """
        print "next point...."
        x = np.matrix(x).astype(floatX)
        y = np.array([y]).astype(floatX)
        for i in range((int)(self.params['num_iterations']*learning_intensity)):
            self.train_function(x, y)
        self.add_to_memory(x, y)

    def add_to_memory(self, x, y):
        self.learnedX.append(x)
        self.learnedY.append(y)

    def learn_a(self, X, Y, learning_intensity):
        X = np.array(X).astype(floatX)
        Y = np.array(Y).astype(floatX)
        X = np.matrix(X).T
        Y = np.array(Y)
        for i in range((int)(self.params['num_iterations']*learning_intensity)):
            self.train_function(X, Y)

    def make_pred_in_one_point(self, X):
        # получает точку входного пространства
        # возвращает ответ сети в этой точке
        X = np.array(X).astype(floatX)
        X = np.matrix(X).T
        x = theano.tensor.matrix('x', dtype=theano.config.floatX)
        prediction = lasagne.layers.get_output(self.model, x, deterministic=False)
        f = theano.function([x], prediction)
        output = f(X)
        return output

    def save_model_to_file(self, filename):
        my_dict = {'input_var': self.input_var, 'target_var': self.target_var, 'model': self.model, 'params': lasagne.layers.get_all_param_values(self.model)}
        pickle.dump(my_dict, open(filename + ".pkl", "wb"))

    def restore_model_from_file(self, filename):
        my_dict = pickle.load(open(filename + ".pkl", "rb"))
        self.model = my_dict['model']
        self.input_var = my_dict['input_var']
        self.target_var = my_dict['target_var']
        lasagne.layers.set_all_param_values(self.model, my_dict['params'])

    def save_info_to_file(self, filename):
        my_dict = {'model_params': self.params, 'learnedX': self.learnedX, 'learnedY':self.learnedY}
        pickle.dump(my_dict, open(filename + ".pkl", "wb"))

    @staticmethod
    def get_info_from_file(filename):
        my_dict = pickle.load(open(filename + ".pkl", "rb"))
        return my_dict['model_params'], my_dict['learnedX'], my_dict['learnedY']

if __name__ == "__main__":
    import visualisator
    import matplotlib.pyplot as plt
    regressor = DropoutRegressor()


    x = [0, 1]
    y = [1, 2]
    regressor.learn_a(x, y, 1)
    regressor.learn_a(x, y, 1)
    regressor.learn_a(x, y, 1)
    regressor.learn_a(x, y, 1)

    plt.figure()
    visualisator.draw_process(regressor, -3, 3, 30)
    plt.savefig("G_"  + ".png")













