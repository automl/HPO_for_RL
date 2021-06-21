import numpy as np

class Perturb:
    """
    A simple perturb mechanism as specified in (Jaderberg et al., 2017).
    """
    def __init__(self, cs_space=None, boundaries={}):
        self.boundaries = boundaries
        self.cs_space = cs_space
    
    def __call__(self, hyperparameters: dict) -> dict:
        """
        Perturb the nodes in the input.
        :param hyperparameters: A dict with nodes.
        :return: The perturbed nodes.
        """ 
        result = hyperparameters.copy()

        for key in hyperparameters:
            temp_value = self.cs_space[key]._inverse_transform(result[key])
            temp_value += np.random.choice([-1, 1]) * 0.2 * temp_value
            result[key] = self.cs_space[key]._transform(temp_value)
        self.ensure_boundaries(result)
        return result

    def ensure_boundaries(self, result):
        for key in result:
            if key not in self.boundaries:
                continue
            if result[key] < self.boundaries[key][0]:
                result[key] = self.boundaries[key][0]
            elif result[key] > self.boundaries[key][1]:
                result[key] = self.boundaries[key][1]
