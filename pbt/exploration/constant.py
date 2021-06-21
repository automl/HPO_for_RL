import numpy as np

class Constant:
    """
    A simple constant mechanism which returns the same configuration as given.
    """
    def __init__(self):
        pass
    
    def __call__(self, hyperparameters: dict) -> dict:
        """
        Perturb the nodes in the input.
        :param hyperparameters: A dict with nodes.
        :return: The perturbed nodes.
        """
        result = hyperparameters.copy()
        return result

