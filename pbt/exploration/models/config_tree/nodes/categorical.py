from collections import Counter

import numpy as np

from pbt.exploration.models.config_tree.nodes import Node


class Categorical(Node):
    def __init__(self, name, values):
        self.name = name
        self.keys = list(values.keys())
        self.values = values
        self.probabilities = [1.0/len(self.keys) for _ in self.keys]

    def sample(self, result):
        value = str(np.random.choice(self.keys, p=self.probabilities))
        result[self.name] = value
        if self.values[value] is not None:
            self.values[value].sample(result)

    def uniform_sample(self, result):
        result[self.name] = str(np.random.choice(self.keys))

    def evaluate(self, data, scores):
        single_scores = [
            self._get_log_density(point[self.name]) for point in data]
        for i, single_score in enumerate(single_scores):
            scores += single_score

    def _get_log_density(self, value):
        return np.log(self.probabilities[self.keys.index(value)])

    def fit(self, values):
        counter = Counter(values)
        self.probabilities = [
            counter[key]/len(values) for key in self.keys]
        self._add_random_exploration()

    def _add_random_exploration(self, random_probability=0.1):
        factor = 1.0 - random_probability
        addend = random_probability / len(self.probabilities)

        for i, _ in enumerate(self.probabilities):
            self.probabilities[i] *= factor
            self.probabilities[i] += addend

    def get_children(self):
        result = []
        for child in self.values.values():
            if child is not None:
                result.append(child)
                result += child.get_children()
        return result

    def structural_copy(self):
        values = {
            name: node.structural_copy() if node is not None else None
            for name, node in self.values.items()}
        return Categorical(self.name, values)
