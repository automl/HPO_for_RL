import random
import numpy as np
from pbt.exploration import Resample
import itertools

class Exploration_BT():
    def __init__(
            self, mutations: dict, cs_space: dict, resample_probability: float = 0.25,
            boundaries={}):
        """
        A strategy to do both perturb and resample.
        :param mutations: A dictionary with hyperparameter names and mutations
        :param resample_probability: The probability to resample for each call
        """
        self.resample_probability = resample_probability
        self.resample = Resample(mutations=mutations)
        self.records = {}
        self.cs_space = cs_space
        self.boundaries = boundaries

    def __call__(self, hyperparameters: dict, model_id: int, model_time_step: int) -> dict:
        """
        Tabu search of the all possible perturbation. If all possibilities are tried, then sample randomly
        from the configuration space
        :param hyperparameters: The nodes to perturb
        :return: Perturbed and probably resampled nodes
        """
        result = hyperparameters.copy()
        num_hyperparameters = len(result)
        if (model_id, model_time_step) not in self.records:
            self.records[(model_id, model_time_step)] = list(itertools.product([-1,1], repeat=num_hyperparameters))
            random.shuffle(self.records[(model_id, model_time_step)])

        if random.random() < self.resample_probability or len(self.records[(model_id, model_time_step)]) == 0:
            result = self.resample(result)
        else:
            result = self.perturb(result, model_id, model_time_step)

        return result
    
    def perturb(self, hyperparameters: dict, model_id: int, model_time_step: int) -> dict:
        directions = self.records[(model_id, model_time_step)].pop()
        for i, key in enumerate(sorted(hyperparameters)):
            temp_value = self.cs_space[key]._inverse_transform(hyperparameters[key])
            temp_value += directions[i] * 0.2 * temp_value
            hyperparameters[key] = self.cs_space[key]._transform(temp_value)
        self.ensure_boundaries(hyperparameters)
        return hyperparameters

    def ensure_boundaries(self, result):
        for key in result:
            if key not in self.boundaries:
                continue
            if result[key] < self.boundaries[key][0]:
                result[key] = self.boundaries[key][0]
            elif result[key] > self.boundaries[key][1]:
                result[key] = self.boundaries[key][1]
