import abc


class Node:
    def __init__(self, name):
        self.name = name
        self.index = None

    @abc.abstractmethod
    def sample(self, result):
        raise NotImplementedError

    @abc.abstractmethod
    def uniform_sample(self, result):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, data, scores):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, values):
        raise NotImplementedError

    @abc.abstractmethod
    def get_children(self):
        raise NotImplementedError

    @abc.abstractmethod
    def structural_copy(self):
        raise NotImplementedError
