class ConfigTree:
    def __init__(self, root):
        self.root = root
        self.all_nodes = self._get_all_nodes()
        self.hyperparameter_names = self._get_hyperparameter_names()
        self._distribute_indices()

    def sample(self):
        result = self._sample_from_root_nodes()
        for hyperparameter in self.hyperparameter_names:
            if hyperparameter not in result:
                result[hyperparameter] = None
        return result

    def uniform_sample(self):
        result = {}
        for node in self.all_nodes:
            node.uniform_sample(result)
        return result

    def evaluate(self, data):
        scores = [0.0 for _ in data]
        for node in self.root:
            node.evaluate(data, scores)
        return scores

    def fit(self, data):
        for node in self.all_nodes:
            node.fit([point[node.index] for point in data])

    def structural_copy(self):
        return ConfigTree([node.structural_copy() for node in self.root])

    def _distribute_indices(self):
        for i, node in enumerate(self.all_nodes):
            node.index = i + 3  # Place 0, 1, and 2 are score and time_step ...

    def _sample_from_root_nodes(self):
        result = {}
        for node in self.root:
            node.sample(result)
        return result

    def _get_all_nodes(self):
        all_nodes = []
        for node in self.root:
            all_nodes.append(node)
            all_nodes += node.get_children()
        return sorted(all_nodes, key=lambda n: n.name)

    def _get_hyperparameter_names(self):
        return [node.name for node in self.all_nodes]
