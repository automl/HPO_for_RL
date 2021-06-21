from pbt.population import Member


class Population:
    def __init__(self, size, stop, results_path, mode='asynchronous', backtracking=False, elite_ratio=0.1):
        self.members = [Member(i, stop) for i in range(size)]
        self.mode = mode
        self.results_path = results_path
        self.backtracking = backtracking
        self.elite_size = max(round(size * elite_ratio), 1)

    def get_next_member(self):
        time_steps = [member.time_step for member in self.members]
        for time_step in range(min(time_steps), max(time_steps) + 1):
            for member in self.members:
                if member.is_free and member.time_step == time_step:
                    return member
            else:
                if self.mode == 'synchronous':
                    return None

    def get_scores(self):
        return {
            member.member_id: member.get_last_score()
            for member in self.members}
    
    def get_scores_by_time_step(self, time_step):
        scores = {}
        for member in self.members:
            member_score = member.get_score_by_time_step(time_step)
            if member_score:
                scores[member.member_id] = member_score
        return scores

    def get_hyperparameters(self, member_id):
        return self.members[member_id].get_hyperparameters()
    
    def get_hyperparameters_by_time_step(self, member_id, time_step):
        return self.members[member_id].get_hyperparameters_by_time_step(time_step)

    def get_latest_time_step(self, model_id):
        return self.members[model_id].time_step
    
    def get_min_time_step(self):
        return min([member.time_step for member in self.members])

    def save_trial(self, trial):
        self.members[trial.member_id].assign_trial(
            trial, self._get_last_score(trial))

    def _get_last_score(self, trial):
        member_of_model = self.members[trial.model_id]
        if member_of_model.time_step == 0:
            return 0.0
        else:
            return member_of_model.get_last_score()

    def update(self, member_id, score):
        current_member = self.members[member_id]
        trial = current_member.save_score(score)
        current_member.log_last_result(self.results_path)
        if current_member.is_done and not current_member.max_time_step:
            self._set_max_time_step(current_member.time_step)
        return trial

    def is_done(self):
        return all(member.is_done for member in self.members)

    def _set_max_time_step(self, max_time_step):
        for member in self.members:
            member.max_time_step = max_time_step

    def get_elites(self):
        # For PBT-BT only, return empty list if it's not in the PBT-BT mode
        if not self.backtracking:
            return []
        else:
            best_trials = self.get_best_trials()
            return best_trials[:self.elite_size]
            # return sorted(best_trials, key=lambda trial: trial.score, reverse=True)[:self.elite_size]
    
    # def get_elites_by_delta_t(self, delta_t):
    #     # For PBT-BT only, return empty list if it's not in the PBT-BT mode
    #     if not self.backtracking:
    #         return []
    #     else:
    #         best_trials = self.get_best_trials_by_delta_t(delta_t)
    #         return sorted(best_trials, key=lambda trial: trial.score, reverse=True)[:self.elite_size]

    def get_best_trials(self):
        return [member.get_best_trial() for member in self.members if member.get_best_trial() is not None]
    
    # def get_best_trials_by_delta_t(self, delta_t):
    #     return [member.get_best_trial_by_delta_t(delta_t) for member in self.members]