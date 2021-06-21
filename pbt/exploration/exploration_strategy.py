from abc import abstractmethod


class ExplorationStrategy:
    @abstractmethod
    def __call__(self, hyperparameters):
        """
        This method should implement the exploration behaviour.
        :param hyperparameters: The nodes to explore
        :return: The changed nodes
        """
        raise NotImplementedError('This method has to be overwritten!')

    def update(self, trial):
        pass
