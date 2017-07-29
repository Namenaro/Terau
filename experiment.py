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
    data = data_generator.AlexData(15)
    X, Y = data.get_batch(15)
    plt.scatter(X, Y, c='k', label='data', zorder=1)
    plt.savefig("dataset.png")
    data_generator.DataSaver.save_XY_to_file(X, Y, "dataset")

def experiment1():
    num_points = 55
    dg = data_generator.AlexData(num_points)
    X, Y = dg.get_batch(num_points)
    model = dropout_regressor.DropoutRegressor()
    model_name = "model_before"
    info_name = "info_before"
    model.save_info_to_file(info_name)
    model.save_model_to_file(model_name)
    visualise_model(model_name, info_name)
    for i in range(num_points):
        x = X[i]
        y = Y[i]
        model.learn(x,y, learning_intensity=0.5)
    model_name = "model_1"
    info_name = "info_1"
    model.save_info_to_file(info_name)
    model.save_model_to_file(model_name)
    visualise_model(model_name, info_name)
    for i in range(num_points):
        x = X[i]
        y = Y[i]
        model.learn(x,y, learning_intensity=0.5)
    model_name = "model_2"
    info_name = "info_2"
    model.save_info_to_file(info_name)
    model.save_model_to_file(model_name)
    visualise_model(model_name, info_name)
    for i in range(num_points):
        x = X[i]
        y = Y[i]
        model.learn(x,y, learning_intensity=0.5)

    model_name = "model_3"
    info_name = "info_3"
    model.save_info_to_file(info_name)
    model.save_model_to_file(model_name)
    visualise_model(model_name, info_name)


def visualise_model(model_file, info_file):
    P, X, Y = dropout_regressor.DropoutRegressor.get_info_from_file(info_file)
    model = dropout_regressor.DropoutRegressor(file_with_model=model_file)
    plt.figure()
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=15)
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=15)
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=15)
    visualisator.draw_trajectory(model, from_=-1.5, to_=1.5, steps=15)
    plt.scatter(X, Y, c='r', label='real_data', zorder=1)
    plt.savefig(model_file + "Result.png")

if __name__ == "__main__":
    in_lab("experiment6", experiment1)
    



