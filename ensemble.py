# -*- coding: utf-8 -*
import dropout_regressor

class Ensemble:
    def __init__(self):
        self.errors = []
        self.units = {}
        self.units[0] = dropout_regressor.DropoutRegressor()
        self.units[1] = dropout_regressor.DropoutRegressor()

    def _get_unsertainy(self, N, variance):
        return variance/float(N)

    def get_means_unses(self, x):
        predictive_means = {}
        unsertainties = {}
        for key, unit in self.units.items():
            mean, var = unit.predict(x)
            predictive_means[key] = mean
            unsertainties[key] =  self.get_unsertainy(N=unit.N, variance=var)
        return predictive_means, unsertainties

    def get_intensities(self, y, predictive_means, unsertainties):
        intensities = {}
        return intensities


    def feed(self, x, y):
        predictive_means, unsertainties = self.get_means_unses(x)
        intensities = self.get_intensities(y, predictive_means, unsertainties)
        for key, intensity in intensities():
            self.units[key].learn(x, y, intensity)



if __name__ == "__main__":
    pass

