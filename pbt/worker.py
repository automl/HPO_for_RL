from config_space.config_space import DEFAULT_CONFIGSPACE
import numpy as np

import logging
import os
from time import sleep

import Pyro4
import json

from pbt.network import WorkerDaemon, ControllerAdapter, CONTROLLER_URI_FILENAME
from pbt.population import Trial
from pbt.tqdm_logger import TqdmLoggingHandler
from tqdm import tqdm
from scipy.io import savemat


class Criterion:
    def __init__(self, criterion_mode, **kwargs):
        self.criterion_mode = criterion_mode
        self.kwargs = kwargs

    def __call__(self, traj_rets, traj_eval_rets=None, info=None):
        # Concatenate train and eval returns
        if traj_eval_rets is None:
            all_rets = traj_rets
        else:
            all_rets = np.concatenate([traj_rets, traj_eval_rets], axis=1)
        if self.criterion_mode == "max":
            # Take the maximal score over the past
            score = np.mean(all_rets, axis=1).max().item()
        elif self.criterion_mode == 'mean':
            # Take the average score over the past
            score = np.mean(all_rets, axis=1).mean().item()
        elif self.criterion_mode == 'lastk':
            last_k = self.kwargs.get('last_k', 1)
            score = np.mean(all_rets, axis=1)[-last_k:].mean().item()
        #elif self.criterion_mode == 'weighted_return':
        #    
        else:
            raise NotImplementedError("%s is an invalid criterion mode" %self.criterion_mode)
        return score

class PBTWorker:
    # TODO: Doc, worker that sequentially calls step and evaluate
    def __init__(self, worker_id, agent, policy_constructor, train_func, criterion_mode='mean', data_path=os.getcwd(),
                    wait_time=5, initial_step=4, step=1, not_copy_data=False, **kwargs):
        self.logger = logging.getLogger('pbt')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(TqdmLoggingHandler())
        self.worker_id = worker_id
        self.agent = agent
        self.policy_constructor = policy_constructor
        self.train_func = train_func
        self.data_path = data_path
        self.wait_time = wait_time
        self.initial_step = initial_step
        self.step = step
        # TODO: _is_done
        self.is_done = False
        self._controller = None
        self.not_copy_data = not_copy_data
        # Ensure safety of criterion mode lastk
        if self.not_copy_data:
            self.logger.info('NOTICE: We are not copying data in this experiment.')
        if kwargs.get('last_k', 0) > initial_step + 1:
            raise ValueError("criterion of last k must have at least same as initial_step + 1")
        self.criterion = Criterion(criterion_mode, **kwargs)

    def register(self, controller=None):
        if controller:
            self.logger.info('Registered controller directly.')
            self._controller = controller
            self._controller.register_worker(self)
        else:
            self.logger.info('Registered controller over network.')
            self._controller = ControllerAdapter(self._discover_controller())
            self._run_daemon()

    def _run_daemon(self):
        self.logger.info('Starting worker daemon.')
        daemon = WorkerDaemon(self)
        uri = daemon.start()
        success = self._controller.register_worker_by_uri(uri)
        if not success:
            daemon.stop()
            raise Exception(f'The read controller URI "{uri}" is not valid!')

    def run(self):
        if not self._controller:
            self.register()

        while not self.is_done:
            self._run_iteration()

    def stop(self):
        self.logger.info('Shutting down worker.')
        self.is_done = True

    def _run_iteration(self):
        trial = self._load_trial()
        if not trial:
            return

        hyperparameters = trial.hyperparameters
        policy = self.policy_constructor(hyperparameters, DEFAULT_CONFIGSPACE)
        if trial.time_step == 0:
            traj_obs, traj_acs, traj_rets, traj_rews, info = self.train_func(self.agent, policy, step=self.initial_step)
        else:
            path = self._get_last_model_path(trial)
            policy.load_model(path)
            traj_obs, traj_acs, traj_rets, traj_rews = self.load_training_data(path)
            traj_obs, traj_acs, traj_rets, traj_rews, info = self.train_func(self.agent, policy,
                traj_obs, traj_acs, traj_rets, traj_rews, step=self.step)
        score = self.criterion(traj_rets, info=info)

        save_path = self._create_model_path(trial)
        policy.save_model(save_path)
        self.save_training_data(save_path, traj_obs, traj_acs, traj_rets, traj_rews)
        self.save_infomation(save_path, info)
        self._send_evaluation(trial.member_id, score)
    
        
    def load_training_data(self, path):
        with open(os.path.join(path, "traj_obs.json"), 'r') as f:
            traj_obs = json.load(f)
        with open(os.path.join(path, "traj_acs.json"), 'r') as f:
            traj_acs = json.load(f)
        with open(os.path.join(path, "traj_rets.json"), 'r') as f:
            traj_rets = json.load(f)
        with open(os.path.join(path, "traj_rews.json"), 'r') as f:
            traj_rews = json.load(f)

        return traj_obs, traj_acs, traj_rets, traj_rews

    def save_training_data(self, path, traj_obs, traj_acs, traj_rets, traj_rews):
        with open(os.path.join(path, "traj_obs.json"), 'w') as f:
            json.dump(traj_obs, f)
        with open(os.path.join(path, "traj_acs.json"), 'w') as f:
            json.dump(traj_acs, f)
        with open(os.path.join(path, "traj_rets.json"), 'w') as f:
            json.dump(traj_rets, f)
        with open(os.path.join(path, "traj_rews.json"), 'w') as f:
            json.dump(traj_rews, f)

    def save_infomation(self, path, info):
        savemat(
            os.path.join(path, "infos.mat"),
            {
               key : info[key] for key in info.keys()
            },
            long_field_names=True
        )

    def _load_trial(self):
        trial = Trial.from_tuple(*self._get_trial())

        if not trial.is_valid():
            self.logger.info(
                f'No trial ready. Waiting for {self.wait_time} seconds.')
            sleep(self.wait_time)
        else:
            self.logger.info(f'Got valid trial {trial} from controller.')
            return trial

    def _discover_controller(self):
        self.logger.debug('Discovering controller.')
        file_path = os.path.join(self.data_path, CONTROLLER_URI_FILENAME)
        tqdm.write(file_path)
        from time import sleep
        sleep(5)
        for number_of_try in range(5):
            try:
                with open(file_path, 'r') as f:
                    uri = f.readline().strip()
                break
            except FileNotFoundError:
                self.logger.info('Can\'t reach controller. Waiting ...')
                sleep(5)
                if number_of_try < 4:
                    continue
            raise Exception('Can\'t reach controller!')
        return Pyro4.Proxy(uri)

    def _get_trial(self):
        # TODO: Intercept connection issues
        return self._controller.request_trial()

    def _get_last_model_path(self, trial):
        # TODO: Remove ambiguity with model_id <-> member_id
        if self.not_copy_data:
            path = os.path.join(
                self.data_path, str(trial.member_id), str(trial.model_time_step))
        else:
            path = os.path.join(
                self.data_path, str(trial.model_id), str(trial.model_time_step))
        return path
            
    def _create_model_path(self, trial):
        path = os.path.join(
            self.data_path, str(trial.member_id), str(trial.time_step))
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
    
    def _get_last_model_path(self, trial):
        path = os.path.join(
            self.data_path, str(trial.member_id), str(trial.model_time_step))
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _create_extra_data_path(self, trial):
        path = os.path.join(
            self.data_path, 'extra', str(trial.member_id), str(trial.time_step))
        if not os.path.isdir(path):
            os.makedirs(path)
        return path

    def _send_evaluation(self, member_id, score):
        # TODO: Handle connection issues
        self.logger.info(f'Sending evaluation. Score: {score}')
        self._controller.send_evaluation(member_id, float(score))
