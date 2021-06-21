import random

from pbt.exploration import ExplorationStrategy, Perturb, Resample


class PerturbAndResample(ExplorationStrategy):
    def __init__(
            self, mutations: dict, cs_space: dict, resample_probability: float = 0.25,
            boundaries={}):
        """
        A strategy to do both perturb and resample.
        :param mutations: A dictionary with hyperparameter names and mutations
        :param resample_probability: The probability to resample for each call
        """
        self.resample_probability = resample_probability

        self.perturb = Perturb(cs_space=cs_space, boundaries=boundaries)
        self.resample = Resample(mutations=mutations)

    def __call__(self, hyperparameters: dict) -> dict:
        """
        Perturb all nodes specified by mutations and then resample
        each hyperparameter depending on the resample_probability.
        :param hyperparameters: The nodes to perturb
        :return: Perturbed and probably resampled nodes
        """
        result = self.perturb(hyperparameters)

        if random.random() < self.resample_probability:
            result = self.resample(result)

        return result
