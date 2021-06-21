import json
import os


class Member:
    def __init__(self, member_id, stop):
        self.member_id = member_id
        self.stop = stop

        self.trials = {}
        self.time_step = 0
        self.max_time_step = None
        self.last_score = 0
        self._actual_time_step = 0

        self.is_free = True
        self.is_done = False

    def __repr__(self):
        return f'<Member id={self.member_id}, time_step={self.time_step}>'

    def assign_trial(self, trial, last_score):
        self.is_free = False
        self.trials[trial.time_step] = trial
        self.last_score = last_score

    def reached_time_step(self, time_step):
        if time_step not in self.trials:
            return False
        return self.trials[time_step].score is not None

    def get_last_score(self):
        if self.time_step == 0:
            return -float('inf')
        return self.trials[self.time_step - 1].score
    
    def get_score_by_time_step(self, time_step):
        if self.time_step == 0:
            return -float('inf')
        if time_step not in self.trials:
            return None
        return self.trials[time_step].score
    
    def get_last_trial(self):
        return self.trials[self.time_step - 1]
    
    def get_all_scores(self):
        return [trial.score for trial in self.trials]

    def get_best_trial(self):
        valid_trials = [trial for trial in self.trials.values() if trial.score is not None]
        if len(valid_trials) == 0:
            return None
        else:
            return max(valid_trials, key=lambda trial: trial.score)

    # def get_best_trial_by_delta_t(self, delta_t):
    #     return sorted(self.trials[::delta_t], key=lambda trial: trial.score)[-1]
    
    def get_hyperparameters_by_time_step(self, time_step):
        if time_step not in self.trials:
            return None
        return self.trials[time_step].hyperparameters

    def get_hyperparameters(self):
        return self.trials[self.time_step - 1].hyperparameters

    def save_score(self, score):
        trial = self.trials[self.time_step]
        trial.score = score
        trial.improvement = score - self.last_score
        self.time_step += 1
        self._actual_time_step += 1

        # TODO: Move this to own function
        if self.max_time_step:
            if self._actual_time_step >= self.max_time_step:
                self.is_done = True
            else:
                self.is_free = True
        else:
            if self.stop(self._actual_time_step, score):
                self.is_done = True
            else:
                self.is_free = True

        return trial
    
    def set_actual_time_step(self, time_step):
        self._actual_time_step = time_step

    def get_actual_time_step(self):
        return self._actual_time_step
    

    def log_last_result(self, results_path):
        last_result = self.trials[self.time_step - 1]
        member_path = os.path.join(results_path, str(self.member_id))
        step_path = os.path.join(member_path, str(last_result.time_step))
        if not os.path.isdir(step_path):
            os.makedirs(step_path)
        with open(os.path.join(member_path, 'scores.txt'), 'a+') as f:
            f.write(f'{last_result.score}\n')
        with open(os.path.join(step_path, 'nodes.json'), 'w') as f:
            json.dump(last_result.hyperparameters, f)
        with open(os.path.join(step_path, 'add.json'), 'w') as f:
            json.dump(
                {'copied from': self.trials[self.time_step - 1].model_id,
                 'time step:': self.trials[self.time_step - 1].model_time_step}, f)

    def _create_hyperparameters(self):
        if self.time_step == 0:
            return self.exploration.get_start_hyperparameters()
        return self.exploration()
