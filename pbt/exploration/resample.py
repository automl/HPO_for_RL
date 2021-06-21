import random


class Resample:
    def __init__(self, mutations: dict):
        """
        A simple resample mechanism as specified in (Jaderberg et al., 2017).
        :param mutations: A dict of all nodes and its mutations.
        """
        self.mutations = mutations

    def __call__(self, hyperparameters: dict) -> dict:
        """
        Resample nodes given by the specified mutations.
        :param hyperparameters: All nodes
        :return: All nodes with specified nodes resampled
        """
        result = hyperparameters.copy()

        for key, value in self.mutations.items():
            result[key] = value() if callable(value) else random.choice(value)

        return result
