# -*- coding: utf-8 -*
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(43)

# Класс отвечающий за датасет для регресии из 1д в 1д.
# Класс инициализирует датасет
# Выдает по запросу случайное кол-во семплоов
# отрисовывает датасет
class AlexData:
    def __init__(self, size):
        self.X = []
        self.Y = []
        self.size = size
        self.make_XY(size)

    def _make_Y(self, X, noise, koef=1):
        Y = []
        for x in X:
            y = koef * np.sin(x) + noise * np.random.normal()
            Y.append(y)
        return Y

    def _make_X(self, mu, sigma, n):
        X = []
        for _ in range(n):
            x = np.random.normal(mu, sigma)
            X.append(x)
        return X

    def XY(self, num_samples):
        n1 = int(num_samples)
        #n2 = num_samples - n1
        X1 = self._make_X(mu=0.0, sigma=0.3, n=n1)
        #X2 = self._make_X(mu=1.5, sigma=0.3, n=n2)
        Y1 = self._make_Y(X1, noise=0.0, koef=1)
        #Y2 = self._make_Y(X2, noise=0.00, koef=-0.5)
        X = X1
        Y = Y1
        return X, Y

    def make_XY(self, num_samples):
        X, Y = self.XY(num_samples)
        self.X = X
        self.Y = Y

    def show_XY(self, X=None, Y=None):
        if X is None:
            X = self.X
            Y = self.Y
        plt.figure()
        print str(X)
        print str(Y)
        plt.scatter(X, Y, c='k', label='data', zorder=1)
        plt.show()

    def get_batch(self, size=None):
        if size is None:
            size = self.size
        assert size <= len(self.X)
        assert len(self.X) == len(self.Y)
        indexes = np.random.choice(len(self.X), size)
        batchX = []
        batchY = []
        for index in indexes:
            batchX.append(self.X[index])
            batchY.append(self.Y[index])
        return batchX, batchY

    def get_test_data(self, size):
        X, Y = self.XY(size)
        return X, Y


class TwoGroups:
    X = [-0.6, -0.7, -0.8, -1,  0.5, 0.6, 0.65, 0.7, 0.75, 0.9]
    Y = [0.1,   0.4,  0.5, 0.6, -1,  -1.5, -2, -2.5, -2.7,  -3]

    @staticmethod
    def get_2_groups():
        return TwoGroups.X, TwoGroups.Y

    @staticmethod
    def get_random_point():
        N = len(TwoGroups.X)
        index = np.random.choice(N)
        return TwoGroups.X[index], TwoGroups.Y[index]


class DataSaver:
    @staticmethod
    def save_XY_to_file(X, Y, filename):
        my_dict = {'X': X, 'Y': Y}
        pickle.dump(my_dict, open(filename + ".pkl", "wb"))

    @staticmethod
    def get_XY_from_file(filename):
        my_dict = pickle.load(open(filename + ".pkl", "rb"))
        return my_dict['X'], my_dict['Y']

if __name__ == "__main__":
     data = AlexData(10)
     data.show_XY()
     #X,Y = data.get_batch(10)
     X, Y = TwoGroups.get_2_groups()
     data.show_XY(X=X, Y=Y)
     x, y = TwoGroups.get_random_point()
     print str(x) + ', '+ str(y)
     x, y = TwoGroups.get_random_point()
     print str(x) + ', ' + str(y)
     x, y = TwoGroups.get_random_point()
     print str(x) + ', ' + str(y)
