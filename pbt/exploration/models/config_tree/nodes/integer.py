import numpy as np
from sklearn.neighbors import KernelDensity

from pbt.exploration.models.config_tree.nodes import Node


class Integer(Node):
    def __init__(self, name, low, high, width=20):
        self.low = low
        self.high = high
        self.kde = KernelDensity((high - low) / width)

        self.kde.fit(np.array([high-low])[:, None])

        super().__init__(name)

    def sample(self, result):
        value = float('inf')
        while value < self.low or value > self.high:
            value = int(self.kde.sample())
        result[self.name] = value

    def uniform_sample(self, result):
        result[self.name] = int(np.round(
            np.random.uniform(low=self.low, high=self.high)))

    def evaluate(self, data, scores):
        single_scores = self.kde.score_samples(
            np.array([point[self.name] for point in data])[:, None])
        for i, single_score in enumerate(single_scores):
            scores[i] += single_score

    def fit(self, values):
        self.kde.fit(np.array(values)[:, None])

    def get_children(self):
        return []

    def structural_copy(self):
        return Integer(self.name, self.low, self.high)
