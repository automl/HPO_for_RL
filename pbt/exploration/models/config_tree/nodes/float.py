from bokeh.io import show
from bokeh.plotting import Figure

import numpy as np
from sklearn.neighbors import KernelDensity

from pbt.exploration.models.config_tree.nodes import Node


class Float(Node):
    def __init__(self, name, low, high, width=20):
        self.low = low
        self.high = high

        self.kde = KernelDensity((high - low) / width)

        self.kde.fit(np.array([high-low])[:, None])

        super().__init__(name)

    def sample(self, result):
        value = float('inf')
        while value < self.low or value > self.high:
            value = float(self.kde.sample())
        result[self.name] = value

    def uniform_sample(self, result):
        result[self.name] = float(np.random.uniform(
            low=self.low, high=self.high))

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
        return Float(self.name, self.low, self.high)


if __name__ == '__main__':
    node = Float('lr', low=1e-5, high=1e-3)
    node.fit([0.0005, 1e-5, 1e-5])
    results = []
    for i in range(100000):
        r = {}
        node.sample(r)
        results.append(r['lr'])

    hist, edges = np.histogram(results, bins=100)

    plot = Figure()
    plot.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:])
    show(plot)
