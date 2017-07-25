# -*- coding: utf-8 -*
class Regressor:
    # интерфейс для модели регрессора
    def __init__(self):
        pass

    def predict(self, x):
        """
        Скормить точку в модель и получить из нее предсказание и дисперсию
        :param point: точка данных
        :return: predictive_mean, predictive_var
        """
        raise NotImplementedError("Implement predict to regressor")

    def learn(self, x, y, learning_intensity):
        """
        Провести итерацию обучения регрессора по новой точке/ах (x, y) данных
        :param x:
        :param y:
        :param learning_intensity:
        """
        raise NotImplementedError("Implement learn to regressor")

