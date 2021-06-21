import logging

from pbt.population import NoTrial, Trial
from pbt.tqdm_logger import TqdmLoggingHandler

class Scheduler:
    def __init__(
            self, population, start_hyperparameters, exploitation, exploration):
        self.logger = logging.getLogger('pbt')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(TqdmLoggingHandler())
        self.population = population
        self.exploitation = exploitation
        self.exploration = exploration
        self.start_hyperparameters = start_hyperparameters

    def get_trial(self):
        self.logger.debug('Trial requested.')
        member = self.population.get_next_member()
        if not member:
            self.logger.debug('No trial ready.')
            return NoTrial()

        if member.time_step == 0:
            start_hyperparameters = {cfg_name : self.start_hyperparameters[cfg_name]()
                for cfg_name in self.start_hyperparameters.keys()}
            trial = Trial(
                member.member_id, -1, 0, -1, start_hyperparameters)
            self.population.save_trial(trial)
            self.logger.debug(f'Returning first trial {trial}.')
            return trial

        self.logger.debug(f'Generating trial for member {member.member_id}.')
        scores = self.population.get_scores_by_time_step(member.time_step - 1)
        # model_id indicates the model that we want to copy
        model_id = self.exploitation(member.member_id, scores)
        # model_time_step = self.population.get_latest_time_step(model_id) - 1
        model_time_step = member.time_step - 1
        hyperparameters = self.population.get_hyperparameters_by_time_step(model_id, model_time_step)
        if model_id != member.member_id:
            self.logger.debug(f'Copying model {model_id}.')
            hyperparameters = self.exploration(hyperparameters)
            self.logger.debug(f'Using exploration. New: {hyperparameters}')
        else:
            self.logger.debug(f'Staying with current model {model_id}.')
        trial = Trial(
            member.member_id, model_id, member.time_step, model_time_step,
            hyperparameters)
        self.population.save_trial(trial)
        return trial

    def update_exploration(self, trial):
        self.exploration.update(trial)
