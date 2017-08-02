# -*- coding: utf-8 -*
import dropout_regressor
import data_generator
import visualisator
import matplotlib.pyplot as plt

class Ensemble:
    def __init__(self, max_size):
        self.max_size = max_size
        self.nodes = {}

    def save(self, damp_name):
        for key, node in self.nodes.items():
            model_name = "mod_" + str(key) + "_" + damp_name
            info_name = "inf_" + str(key) + "_" + damp_name
            node.model.save_info_to_file(info_name)
            node.model.save_model_to_file(model_name)
            visualisator.visualize_dropout_regressor(model_file=model_name, info_file=info_name)

    def _get_errs_and_unserts_in_point(self, x, y):
        predictive_errors = {}
        unsertainties = {}
        for key, node in self.nodes.items():
            mean, var = node.predict(x)
            predictive_errors[key] = self.f_error(mean, y)
            unsertainties[key] = var
        return predictive_errors, unsertainties

    def f_error(self, target, prediction):
        return (target - prediction) ^ 2

    def feed(self, x, y):
        errors, unsertainties = self._get_errs_and_unserts_in_point(x, y)





if __name__ == "__main__":
    ensemble = Ensemble(max_size=4)
    for i in range(100):
        x, y = data_generator.TwoGroups.get_random_point()
        ensemble.feed(x,y)
        if i%10 != 0:
            continue
        ensemble.save("iter_" + str(i))
