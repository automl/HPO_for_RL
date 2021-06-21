class Trial:
    def __init__(
            self, member_id, model_id, time_step, model_time_step,
            hyperparameters):
        self.member_id = member_id
        self.model_id = model_id
        self.time_step = time_step
        self.model_time_step = model_time_step
        self.hyperparameters = self._clean_hyperparameters(hyperparameters)
        self.score = None
        self.improvement = None

    def __repr__(self):
        return f'<Trial ' \
            f'member_id: {self.member_id}, ' \
            f'model_id: {self.model_id}, ' \
            f'time_step: {self.time_step}, ' \
            f'model_time_step: {self.model_time_step}, ' \
            f'nodes: {self.hyperparameters}, ' \
            f'score: {self.score}, ' \
            f'improvement: {self.improvement}>'

    def _clean_hyperparameters(self, hyperparameters):
        result = hyperparameters.copy()
        for key, value in result.items():
            if callable(value):
                result[key] = value()
        return result

    def is_valid(self):
        return True

    def to_tuple(self):
        """
        Return this trial as tuple (easier to send over network).
        :return: (member_id, model_id, time_step, nodes, score)
        """
        return \
            self.member_id, self.model_id, self.time_step, \
            self.model_time_step, self.hyperparameters

    def to_array(self):
        result = [self.score, self.time_step, self.improvement]
        result += [
            self.hyperparameters[key]
            for key in sorted(self.hyperparameters)]
        return result

    @staticmethod
    def from_tuple(
            member_id, model_id, time_step, model_time_step, hyperparameters):
        return Trial(
            member_id, model_id, time_step, model_time_step, hyperparameters) \
            if member_id is not -1 \
            else NoTrial()

    def copy(self):
        return Trial(
            self.member_id, self.model_id, self.time_step, self.model_time_step,
            self.hyperparameters.copy())


class NoTrial(Trial):
    def __init__(self):
        super().__init__(-1, -1, -1, -1, {})

    def is_valid(self):
        return False
