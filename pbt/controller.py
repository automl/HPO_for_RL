import numpy as np

import random
import logging
import os
import sys
from time import sleep

from pbt.exploitation import Truncation
from pbt.exploration import Perturb
from pbt.garbage_collector import GarbageCollector
from pbt.network import ControllerDaemon
from pbt.population import Population
from pbt.scheduler import Scheduler
from pbt.tqdm_logger import TqdmLoggingHandler

class PBTController:
    def __init__(
            self, pop_size, start_hyperparameters, exploitation=Truncation(),
            exploration=Perturb(), ready=lambda: True,
            stop=lambda iterations, _: iterations >= 100.0,
            data_path=os.getcwd(), results_path=os.getcwd(),
            max_steps=sys.maxsize):

        self.logger = logging.getLogger('pbt')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(TqdmLoggingHandler())
        self.daemon = None
        self.population = Population(pop_size, stop, results_path)
        self.data_path = data_path
        self.scheduler = Scheduler(
            self.population, start_hyperparameters, exploitation, exploration)
        self.garbage_collector = GarbageCollector(data_path)
        self.ready = ready
        self.max_steps = max_steps
        self.total_steps = 0

        self.workers = {}
        self._done = False

        self.logger.info(
            f'Started controller with parameters: ' +
            f'population size: {pop_size}, '
            f'data_path: {data_path}, '
            f'results_path: {results_path}')

    def start_daemon(self):
        self.logger.info('Starting daemon.')
        self.daemon = ControllerDaemon(self)
        self.daemon.start()

    def register_worker(self, worker):
        self.logger.debug(f'Worker {worker.worker_id} registered.')
        self.workers[worker.worker_id] = worker

    def request_trial(self):
        trial = self.scheduler.get_trial()
        return trial.to_tuple()

    def send_evaluation(self, member_id, score):
        self.logger.debug(
            f'Receiving evaluation for member {member_id}: {score}')
        # TODO: Some sort of ready function?
        min_timestep = self.population.get_min_time_step()
        # exclude all elites members (PBT-BT only)
        elites = []
        self.logger.info(f'min time step is {min_timestep}')
        self.garbage_collector.collect(member_id, min_timestep, elites)
        trial = self.population.update(member_id, score)
        self.scheduler.update_exploration(trial)
        self.total_steps += 1
        if self.population.is_done():
            self.logger.info('Nothing more to do. Shutting down.')
            self._shut_down_workers()
            sleep(10)
            if self.daemon:
                self.daemon.shut_down()
        
        if self.total_steps >=  self.max_steps:
            self.logger.info('Reach the maximal steps. Shutting down.')
            self._shut_down_workers()
            sleep(10)
            if self.daemon:
                self.daemon.shut_down()

    def _shut_down_workers(self):
        for worker in self.workers.values():
            worker.stop()
        