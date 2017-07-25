# -*- coding: utf-8 -*
import os
import matplotlib.pyplot as plt

import utils
import data_generator
import ensemble
import dropout_regressor
import visualisator

def in_lab(name, experiment):
    oldpwd = os.getcwd()
    utils.setup_folder_for_results(main_folder=name)
    experiment()
    os.chdir(oldpwd)

def experiment(): # сохранение датасета в картинку и в файл
    plt.figure()
    data = data_generator.AlexData(30)
    X, Y = data.get_batch(30)
    plt.scatter(X, Y, c='k', label='data', zorder=1)
    plt.savefig("dataset.png")
    data_generator.DataSaver.save_XY_to_file(X, Y, "dataset")

def experiment1():
    dg = data_generator.AlexData(30)
    model = dropout_regressor.DropoutRegressor()
    for i in range(100):
        X, Y = dg.get_batch(1)
        x = X[0]
        y = Y[0]
        model.learn(x,y, 1)
        if i%9 == 0:
            model.save_info_to_file("info_"+str(i))
            model.save_model_to_file("model"+str(i))


if __name__ == "__main__":
    in_lab("experiment1", experiment)
    



