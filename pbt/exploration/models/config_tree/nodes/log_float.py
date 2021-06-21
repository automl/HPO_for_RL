import numpy as np

from pbt.exploration.models.config_tree.nodes import Float


class LogFloat(Float):
    def __init__(self, name, low, high, width=20):
        if low <= 0:
            raise ValueError('"low" has to be greater than 0!')
        super().__init__(name, low, high, width)

    def sample(self, result):
        value = float('inf')
        while value < self.low or value > self.high:
            value = float(10**self.kde.sample())
        result[self.name] = value

    def uniform_sample(self, result):
        result[self.name] = float(10**np.random.uniform(
            low=np.log10(self.low), high=np.log10(self.high)))
        return result[self.name]

    def evaluate(self, data, scores):
        single_scores = self.kde.score_samples(
            np.array([np.log10(point[self.name]) for point in data])[:, None])
        for i, single_score in enumerate(single_scores):
            scores[i] += single_score

    def fit(self, values):
        self.kde.fit(np.array([np.log10(value) for value in values])[:, None])

    def get_children(self):
        return []

    def structural_copy(self):
        return LogFloat(self.name, self.low, self.high)
