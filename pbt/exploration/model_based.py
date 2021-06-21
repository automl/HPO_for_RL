from pbt.exploration import ExplorationStrategy


class ModelBased(ExplorationStrategy):
    def __init__(self, model):
        self.model = model

    def __call__(self, hyperparameters):
        return self.model.sample()

    def update(self, trial):
        self.model.update(trial.to_array())
