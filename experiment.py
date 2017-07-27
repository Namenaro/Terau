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
    for i in range(10):
        X, Y = dg.get_batch(1)
        x = X[0]
        y = Y[0]
        model.learn(x,y, 10)

    model_name = "model_"
    info_name = "info_"
    model.save_info_to_file(info_name)
    model.save_model_to_file(model_name)
    visualise_model(model_name, info_name)


def visualise_model(model_file, info_file):
    P, X, Y = dropout_regressor.DropoutRegressor.get_info_from_file(info_file)
    model = dropout_regressor.DropoutRegressor(file_with_model=model_file)
    plt.figure()
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=20)
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=20)
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=20)
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=20)
    plt.scatter(X, Y, c='r', label='real_data', zorder=1)
    plt.savefig("Result.png")

if __name__ == "__main__":
    in_lab("experiment2", experiment1)
    



