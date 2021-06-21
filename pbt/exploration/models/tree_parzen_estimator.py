import logging
from collections import defaultdict

import numpy as np

from pbt.exploration.models import Model
from pbt.tqdm_logger import TqdmLoggingHandler

class TreeParzenEstimator(Model):
    def __init__(
            self, config_tree, best_percent=0.2, uniform_percent=0.25,
            sample_size=10, new_samples_until_update=10, window_size=20,
            mode='improvement', split='time_step'):
        self.l_tree = config_tree
        self.g_tree = config_tree.structural_copy()

        self._best_percent = best_percent
        self._uniform_percent = uniform_percent
        self._sample_size = sample_size
        self._window_size = window_size

        self._new_samples_until_update = new_samples_until_update
        self._counter = 0

        self.data = defaultdict(list)

        self.logger = logging.getLogger('pbt')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(TqdmLoggingHandler())
        if mode == 'improvement':
            self._mode = 2
        else:
            self._mode = 0

        if split != 'time_step':
            self._split_function = self._split_data
        else:
            self._split_function = self._split_by_time_step

    def sample(self):
        if np.random.random() < self._uniform_percent:
            self.logger.debug('Using random sampling.')
            return self.l_tree.uniform_sample()
        self.logger.debug('TPE sampling:')
        samples = [self.l_tree.sample() for _ in range(self._sample_size)]
        self.logger.debug(f'Samples: {samples}')
        l_scores = self.l_tree.evaluate(samples)
        g_scores = self.g_tree.evaluate(samples)
        scores = [l/g for l, g in zip(l_scores, g_scores)]
        self.logger.debug(f'Scores: {scores}')
        return samples[np.argmin(scores)]

    def update(self, trial):
        self.data[trial[1]].append(trial)

        self._counter += 1
        if self._counter >= self._new_samples_until_update:
            self._fit_data()
            self._counter = 0

    def _fit_data(self):
        indices = range(
            max(0, len(self.data) - self._window_size),
            len(self.data))
        sliding_window = [self.data[x] for x in indices]
        good, bad = self._split_function(sliding_window)
        self.l_tree.fit(good)
        self.g_tree.fit(bad)

    def _split_by_time_step(self, data):
        good, bad = [], []
        for time_step in data:
            new_good, new_bad = self._split_data([time_step])
            good += new_good
            bad += new_bad
        return good, bad

    def _split_data(self, data):
        scores = sorted([
            trial[self._mode]
            for time_step in data for trial in time_step], reverse=True)
        pivot_score = scores[int(len(scores) * self._best_percent)]
        good, bad = [], []
        for time_step in data:
            for point in time_step:
                if point[self._mode] >= pivot_score:
                    good.append(point)
                if point[self._mode] <= pivot_score:
                    bad.append(point)
        return good, bad


