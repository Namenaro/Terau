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

def experiment():
    plt.figure()
    data = data_generator.AlexData(30)
    X, Y = data.get_batch(30)
    plt.scatter(X, Y, c='k', label='data', zorder=1)
    plt.savefig("dataset.png")
    data_generator.DataSaver.save_XY_to_file(X, Y, "dataset")
    newX, newY = data_generator.DataSaver.get_XY_from_file("dataset")
    plt.figure()
    plt.scatter(newX, newY, c='r', label='data', zorder=1)
    plt.savefig("dataset_restored.png")

if __name__ == "__main__":
    name = "Test"
    in_lab(name, experiment)
    



