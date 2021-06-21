import logging

from pbt.population import NoTrial, Trial
from pbt.tqdm_logger import TqdmLoggingHandler
import random
from itertools import product

class BacktrackScheduler:
    def __init__(
            self, population, start_hyperparameters, exploitation, exploration, exploration_bt, delta_t=30, tolerance=0.2):
        """Initialize a scheduler with backtracking mechanism
        """        
        self.logger = logging.getLogger('pbt')
        self.logger.setLevel(logging.DEBUG)
        self.population = population
        self.exploitation = exploitation
        self.exploration = exploration
        self.exploration_bt = exploration_bt
        
        self.start_hyperparameters = start_hyperparameters
        self.delta_t = delta_t
        self.tolerance = tolerance
        self.record = {} # For tabu search

    def get_trial(self):
        self.logger.debug('Trial requested.')
        member = self.population.get_next_member()
        if not member:
            self.logger.debug('No trial ready.')
            return NoTrial()

        # intial run with starting hyperparameter
        if member.time_step == 0:
            start_hyperparameters = {cfg_name : self.start_hyperparameters[cfg_name]()
                for cfg_name in self.start_hyperparameters.keys()}
            trial = Trial(
                member.member_id, -1, 0, -1, start_hyperparameters)
            self.population.save_trial(trial)
            self.logger.debug(f'Returning first trial {trial}.')
            return trial

        if member.time_step % self.delta_t == 0:
            # start to check if it drops by X percentage
            self.logger.debug(f'Generating trial for member {member.member_id} with times step {member.time_step} with BT.')
            self.logger.debug(f'member {member.member_id} actual time step is {member._actual_time_step}.')
            elites = self.population.get_elites()
            model_id, model_time_step = self.backtracking_exploitation(member, elites)
            hyperparameters = self.population.get_hyperparameters_by_time_step(model_id, model_time_step)
            if model_id == member.member_id and model_time_step == member.time_step - 1:
                self.logger.debug(f'Staying with current model {model_id}.')
            else:
                self.logger.debug(f'Backtracking to model {model_id}, time step {model_time_step}.')
                member.set_actual_time_step(model_time_step)
                self.logger.debug(f'Set model {model_id} actual time step to {model_time_step}.')
                # TODO: Replace it with backtracking exploration
                hyperparameters = self.exploration_bt(hyperparameters, model_id, model_time_step)
                self.logger.debug(f'Using exploration. New: {hyperparameters}')
            trial = Trial(
                member.member_id, model_id, member.time_step, model_time_step,
                hyperparameters)
            self.population.save_trial(trial)
            return trial

        # Jointly do standard PBT with elites
        self.logger.debug(f'Generating trial for member {member.member_id} with times step {member.time_step}.')
        self.logger.debug(f'member {member.member_id} actual time step is {member._actual_time_step}.')
        scores = self.population.get_scores()
        # self.logger.debug(f'Collected all scoires.')
        # model_id indicates the model that we want to copy
        model_id = self.exploitation(member.member_id, scores)
        # model_time_step shows the timestep of the model to be copied
        model_time_step = self.population.get_latest_time_step(model_id) - 1
        # actual_time_step = self.population.get_actual_time_step_by_member_id(model_id)
        # self.logger.debug(f'Safely get timestep')
        hyperparameters = self.population.get_hyperparameters_by_time_step(model_id, model_time_step)
        # self.logger.debug(f'Safely get hyperparameters')
        if model_id != member.member_id:
            self.logger.debug(f'Copying model {model_id} at time step {model_time_step}.')
            member.set_actual_time_step(model_time_step)
            self.logger.debug(f'Set model {model_id} actual time step to {model_time_step}.')
            hyperparameters = self.exploration(hyperparameters)
            self.logger.debug(f'Using exploration. New: {hyperparameters}')
        else:
            self.logger.debug(f'Staying with current model {model_id}.')
        trial = Trial(
            member.member_id, model_id, member.time_step, model_time_step,
            hyperparameters)
        self.population.save_trial(trial)
        # self.logger.debug(f'Safely saved trial')
        return trial

    def update_exploration(self, trial):
        self.exploration.update(trial)

    # For PBT-BT only
    def backtracking_exploitation(self, member, elites):
        # TODO: handle zero division
        percentage_change = (member.get_last_score() - elites[-1].score) / abs(elites[-1].score)
        self.logger.debug(f'Performance changed {percentage_change}.')
        if percentage_change < - self.tolerance:
            trial = random.choice(elites)
        else:
            trial = member.get_last_trial()
        
        return trial.member_id, trial.time_step


    
