# -*- coding: utf-8 -*
import dropout_regressor
import data_generator
import visualisator
import matplotlib.pyplot as plt

class Ensemble:
    def __init__(self):
        self.errors = []
        self.units = {}
        self.units[0] = dropout_regressor.DropoutRegressor()
        self.units[1] = dropout_regressor.DropoutRegressor()

    def _get_unsertainy(self, N, variance):
        return variance/float(N)

    def get_means_anses(self, x):
        predictive_means = {}
        unsertainties = {}
        for key, unit in self.units.items():
            mean, var = unit.predict(x)
            predictive_means[key] = mean
            unsertainties[key] = self._get_unsertainy(N=unit.get_N, variance=var)
        return predictive_means, unsertainties

    def get_intensities(self, y, predictive_means, unsertainties):
        intensities = {}
        for key, unsertainty in unsertainties.items():
            err = abs(y - predictive_means[key])
            print "err=" + str(err)
            intensities[key] = 1
        indexes = sorted(unsertainties, key=unsertainties.get)
        intensities[indexes[0]] = err*10
        print "selected " + str(indexes[0])
        return intensities


    def feed(self, x, y):
        predictive_means, unsertainties = self.get_means_anses(x)
        intensities = self.get_intensities(y, predictive_means, unsertainties)
        for key, intensity in intensities.items():
            self.units[key].learn(x, y, intensity)



if __name__ == "__main__":
    data = data_generator.AlexData(30)
    ensemble = Ensemble()
    for i in range(100):
        X, Y = data.get_batch(1)
        x=X[0]
        y=Y[0]
        ensemble.feed(x,y)
        if i%10 != 0:
            continue
        plt.figure()
        visualisator.draw_process(ensemble.units[0], -3, 3, 30)
        plt.savefig("uniT_0_" + str(i) + ".png")
        plt.figure()
        visualisator.draw_process(ensemble.units[1], -3, 3, 30)
        plt.savefig("uniT_1_" + str(i) + ".png")
