import ConfigSpace as CS
from hpbandster.core.worker import Worker
import numpy as np
from config_space.config_space import DEFAULT_CONFIGSPACE

class HPWorker(Worker):
    """"Worker for BOHB/Hyperband/Random Search
    We use the implementation based on https://github.com/automl/HpBandSter
    """
    def __init__(self,
                 train_func,
                 env,
                 config_names,
                 last_k,
                 **kwargs):
        """
        Initialization
        """
        super().__init__(**kwargs)
        self.train_func = train_func
        if len(config_names) == 0:
            raise ValueError("config_names can not be an empty list")
        self.config_names = config_names
        self.env = env
        self.last_k = last_k

    @staticmethod
    def get_configspace(env, config_names):
        """
        Construct the configuration space
        Param:
            env: (str) Name of the environment that we will used

        Return:
            cs: (ConfigurationSpace) A configuration space which defines the search space
        """
        cs = CS.ConfigurationSpace()
        config_lst = []
        for config_name in config_names:
            config_lst.append(DEFAULT_CONFIGSPACE[env][config_name])
        cs.add_hyperparameters(config_lst)
        return cs

    def compute(self, config_id, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.
        Args:
            config_id: (int) configuration id
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        train_func = self.train_func
        # Construct the configurations
        cfg = dict()
        for config_name in DEFAULT_CONFIGSPACE[self.env]:
            if config_name in self.config_names:
                cfg[config_name] = config[config_name]
            else:
                cfg[config_name] = DEFAULT_CONFIGSPACE[self.env][config_name].default_value

        # Run one configuration
        traj_rets, traj_eval_rets = train_func(config_id, cfg, budget)
        all_rets = np.concatenate([traj_rets, traj_eval_rets], axis=1)
        loss = - np.mean(all_rets, axis=1)[-self.last_k:].mean().item()
        return ({
            'loss': loss,
            'info': {}
        })

