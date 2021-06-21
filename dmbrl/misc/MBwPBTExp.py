from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent

from pbt.controller import PBTController
from pbt.worker import PBTWorker
from pbt.exploitation import Truncation
from pbt.exploration import PerturbAndResample

import ConfigSpace.hyperparameters as CSH
from dmbrl.modeling.utils.HPO import build_policy_constructor
from config_space.config_space import DEFAULT_CONFIGSPACE
import pprint
import json
from tqdm import tqdm


class Mutation:
    def __init__(self, cfg_space, random):
        self.cfg_space = cfg_space
        self.random = random
    
    def __call__(self):
        return self.cfg_space.sample(self.random)

class MBWithPBTExperiment:
    activation_fns = {
        "relu": "ReLU",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
        "softmax": "softmax", # Not used
        "swish": "swish"
    }
    model_optimizers = {
        "adam": tf.train.AdamOptimizer,
        "adadelta": tf.train.AdadeltaOptimizer,
        "adagrad": tf.train.AdagradOptimizer,
        "sgd": tf.train.GradientDescentOptimizer,
        "rms": tf.train.RMSPropOptimizer
    }

    def __init__(self, params, cfg_creator, args):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int) (Not Used): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
                .bo_cfg: (Only used in BOHB optimizing)
                    .min_budget (int): Minimal budget to run one configuration.
                        Default to 4.
                    .max_budget (int): Maximal budget to run one configuration.
                        Default to 15.
                    .nopt_iter (int): Number of configurations to be evaluated.
                        Default to 4
                    .eta (int): Base number of Hyperband
                    .n_workers (int): Number of workers to run at the same time.
                        Default to 2

        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        # Set environment seeds
        self.env.seed(args.seed)
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        # self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")
        self.cfg_creator = cfg_creator

        self.log_dir = get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory.")

        self.worker_id = args.worker_id
        self.start_worker = args.worker

        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)

        self.config_names = args.config_names
        self.env_name = args.env

        self.seed = args.seed

        self.criterion_mode = params.pbt_cfg.get("criterion_mode", 'mean')
        if self.criterion_mode not in ['mean', 'lastk', 'max', 'weighted_return ']:
            raise NotImplementedError("Invalid Criterion Mode")
        
        self.last_k = params.pbt_cfg.get("last_k", 1)
        self.initial_step = params.pbt_cfg.get("initial_step", 4)
        self.step = params.pbt_cfg.get("step", 1)
        self.pop_size = params.pbt_cfg.get("pop_size", 10)
        self.budget = params.pbt_cfg.get("budget", 32)

        self.resample_probability = args.resample_probability
        self.sample_from_percent = args.sample_from_percent
        self.resample_if_not_in_percent = args.resample_if_not_in_percent
        self.not_copy_data = args.not_copy_data
        self.control_args = {key: val for (key, val) in args.ctrl_arg}
        self.prop_type = self.control_args['prop-type']
        self.opt_type = self.control_args['opt-type']
        self.translate_into_model = build_policy_constructor(self)

    def run_experiment(self):
        """Perform experiment.
        """
        if self.start_worker:
            self.run_worker()
        else:
            self.run_controller()

    def run_controller(self):
        cs_space = DEFAULT_CONFIGSPACE[self.env_name]

        mutations, start_hyperparameters, boundaries = {}, {}, {}

        self.random = np.random.RandomState(self.seed)
        
        for cfg_name, cfg_space in cs_space.items():
            if cfg_name in self.config_names:
                mutations[cfg_name] = Mutation(cs_space[cfg_name], self.random)
                if type(cfg_space) == CSH.CategoricalHyperparameter:
                    raise NotImplementedError
                boundaries[cfg_name] = (cfg_space.lower, cfg_space.upper)
                start_hyperparameters[cfg_name] = Mutation(cs_space[cfg_name], self.random)
            else:
                pass

        controller = PBTController(
            pop_size=self.pop_size,
            start_hyperparameters=start_hyperparameters,
            exploitation=Truncation(
                sample_from_percent=self.sample_from_percent,
                resample_if_not_in_percent=self.resample_if_not_in_percent),
            exploration=PerturbAndResample(mutations,
                cs_space=cs_space,
                boundaries=boundaries,
                resample_probability=self.resample_probability),
            ready=lambda: True,
            stop=lambda iterations, _: iterations >= self.budget,
            data_path=self.log_dir,
            results_path=self.log_dir)

        tqdm.write("controller starts!")
        controller.start_daemon()

    def run_worker(self):
        worker = PBTWorker(
            worker_id=self.worker_id,
            agent=self.agent,
            policy_constructor=self.translate_into_model,
            criterion_mode=self.criterion_mode,
            train_func=self.train_func,
            data_path=self.log_dir,
            last_k=self.last_k,
            step=self.step,
            initial_step=self.initial_step,
            not_copy_data=self.not_copy_data)
        worker.run()

    def train_func(self,
            agent,
            policy,
            traj_obs=None,
            traj_acs=None,
            traj_rets=None,
            traj_rews=None,
            step=1):
        if traj_obs is None:
            traj_obs = []
        if traj_acs is None:
            traj_acs = []
        if traj_rets is None:
            traj_rets = []
        if traj_rews is None:
            traj_rews = []

        train_loss_lst, test_loss_lst, episode_loss_lst = [], [], []
        for i in range(step):
            samples = []
            # Rollout nrecord episodes
            for j in range(max(self.neval, self.nrollouts_per_iter)):
                samples.append(
                    agent.sample(
                        self.task_hor, policy
                    )
                )
            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.append([sample["reward_sum"] for sample in samples[:self.neval]])
            traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            samples = samples[:self.nrollouts_per_iter]
            if i == 0:
                train_loss, test_loss, episode_loss = policy.train(traj_obs,
                    traj_acs, traj_rews)
            else:
                train_loss, test_loss, episode_loss = policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
            train_loss_lst.append(train_loss)
            test_loss_lst.append(test_loss)
            episode_loss_lst.append(episode_loss)
        info = {
            "train_loss_lst" : train_loss_lst,
            "test_loss_lst" : test_loss_lst,
            "episode_loss_lst" : episode_loss_lst 
        }
        return traj_obs, traj_acs, traj_rets, traj_rews, info

    
