from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time, localtime, strftime, sleep
import pickle

import numpy as np
import tensorflow as tf
import scipy.io as sio
from dotmap import DotMap
# import ds.Schedule
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent
from dmbrl.modeling.layers import FC
from dmbrl.controllers import MPC
import logging
import copy
import itertools
import random
import ConfigSpace.hyperparameters as CSH
from dmbrl.config import create_config
from config_space.config_space import DEFAULT_CONFIGSPACE
import pprint
import json
from tqdm import tqdm
import argparse
from bisect import bisect

def create_cfg_creator(env, ctrl_type, ctrl_args, base_overrides, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    def cfg_creator(additional_overrides=None):
        if additional_overrides is not None:
            return create_config(env, ctrl_type, ctrl_args, base_overrides + additional_overrides, logdir)
        return create_config(env, ctrl_type, ctrl_args, base_overrides, logdir)

    return cfg_creator

def set_random_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class PETS_Trainer:
    activation_fns = {
        "relu": "ReLU",
        "tanh": "tanh",
        "sigmoid": "sigmoid",
        "softmax": "softmax", # Not used
        "swish": "swish"
    }
    model_optimizers = {
        "adam": tf.train.AdamOptimizer,
        # 'adamw': tf.contrib.opt.AdamWOptimizer,
        "adadelta": tf.train.AdadeltaOptimizer,
        "adagrad": tf.train.AdagradOptimizer,
        "sgd": tf.train.GradientDescentOptimizer,
        "rms": tf.train.RMSPropOptimizer
    }

    def __init__(self, args):
        """Initializes class instance.
        Argument:
            args:
        """
        self.env_name = args.env
        self.ctrl_type = args.ctrl_type
        self.ctrl_arg = args.ctrl_arg
        self.override = args.override
        self.logdir = args.logdir
        self.schedule_name = args.schedule_name

        # translate arguments from namespace to a list
        with open(os.path.join("ds", "configs", self.schedule_name+'.json'), 'r') as f:
            self.param_list = json.load(f)
        self.arch_param = {
            "act_idx" : args.act_idx,
            "num_hidden_layers" : args.num_hidden_layers,
            "hidden_layer_width" : args.hidden_layer_width
        }

        self.seed = args.seed
        self.budget = args.budget
        self.last_k = args.last_k
        self.config_id = args.config_id

    def _init_model(self):
        # create cfg_creator
        cfg_creator = create_cfg_creator(self.env_name, self.ctrl_type,
        self.ctrl_arg, self.override, self.logdir)

        # set random seed
        set_random_seed(self.seed)

        cfg = cfg_creator()[0]
        params = cfg.exp_cfg

        env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        env.seed(self.seed)
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            agent = Agent(DotMap(
                env=env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            agent = Agent(DotMap(env=env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        #self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")
        self.cfg_creator = cfg_creator

        self.exp_name = self.env_name + "_" + self.config_id + "_" + str(self.seed)

        parent_dir = get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory.")     
        os.makedirs(parent_dir, exist_ok=True)
        self.log_dir = os.path.join(parent_dir, self.exp_name)
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)

        return agent

    def __call__(self):
        """Perform experiment.
        """
        # Give parameters names to avoid bugs
        agent = self._init_model()
        # Identify if there is pretrained model:
        if os.path.exists(self.log_dir) and os.path.exists(os.path.join(self.log_dir, "training_data.pickle")):
            training_data = self.load_training_data(self.log_dir)
            start_budget = int(training_data["current_budget"]) + 1
            print("Resuming training from %d/%d" %(start_budget+1, self.budget))
        else:
            os.makedirs(self.log_dir, exist_ok=True)
            training_data = None
            start_budget = 0
            print("Starting new training")
        print(self.arch_param)
        if start_budget >= self.budget:
            raise ValueError("Warning! this model has been trained")
        # Training loop
        for i in range(start_budget, self.budget):
            print("Training iter: %d/%d" %(i+1, self.budget))
            print(self.param_list[i])
            policy = self.translate_into_model(self.param_list[i], self.cfg_creator)
            if i > 0:
                policy.load_model(self.log_dir)
            # Reconstruct policy
            training_data = self.train_func(agent, policy, training_data, load_data=True)
            training_data["current_budget"] = i
            if i == self.budget - 1:
                training_data["config"] = self.param_list
            self.save_checkpoint(self.log_dir, policy, training_data)
            print("Rewards obtained:", training_data['traj_rets'][-1])
        score = np.mean(training_data["traj_rets"][-self.last_k:], axis=(0,1)).item()
        print("Training %s finished, score obtained: %f " %(self.exp_name, score))
        self.score = score
        return score

    def train_func(self, agent, policy, training_data=None, load_data=False, step=1):
        if training_data is None:
            training_data = {
                "traj_obs" : [],
                "traj_acs" : [],
                "traj_rets" : [],
                "traj_rews" : [],
                "train_loss_lst" : [],
                "test_loss_lst" : [],
                "episode_loss_lst" : [],
            }

        for i in range(step):
            samples = []
            # Rollout nrecord episodes
            for j in range(max(self.neval, self.nrollouts_per_iter)):
                samples.append(
                    agent.sample(
                        self.task_hor, policy
                    )
                )
            training_data["traj_obs"].extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            training_data["traj_acs"].extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            training_data["traj_rets"].append([sample["reward_sum"] for sample in samples[:self.neval]])
            training_data["traj_rews"].extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            samples = samples[:self.nrollouts_per_iter]
            if load_data:
                train_loss, test_loss, episode_loss = policy.train(
                    training_data["traj_obs"],
                    training_data["traj_acs"],
                    training_data["traj_rews"])
            else:
                train_loss, test_loss, episode_loss = policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
            training_data["train_loss_lst"].append(train_loss)
            training_data["test_loss_lst"].append(test_loss)
            training_data["episode_loss_lst"].append(episode_loss)

        return training_data

    def save_checkpoint(self, path, policy, training_data):
        self.save_training_data(path, training_data)
        policy.save_model(path)

    def load_checkpoint(self, path, policy):
        training_data = self.load_training_data(path)
        policy = policy.load_model(path)
        return policy, training_data
    
    def save_training_data(self, path, training_data):
        with open(os.path.join(path, "training_data.pickle"), 'wb') as handle:
            pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_training_data(self, path):
        with open(os.path.join(path, "training_data.pickle"), 'rb') as handle:
            training_data = pickle.load(handle)
        return training_data

    def translate_into_model(self, param_dict, cfg_creator, reset=True):
        """Translates a parameter vector into a configuration that can be used to construct a policy.
        """
        # Clear the graph to avoid conflicts with previous graphs.
        tf.reset_default_graph()
        config_space = DEFAULT_CONFIGSPACE[self.env_name]
        num_hidden_layers = int(self.arch_param.get("num_hidden_layers", config_space["num_hidden_layers"].default_value))
        hidden_layer_width = int(self.arch_param.get("hidden_layer_width", config_space["hidden_layer_width"].default_value))
        act_idx = self.arch_param.get("act_idx", config_space["act_idx"].default_value)
        model_opt_idx = param_dict.get("model_opt_idx", config_space["model_opt_idx"].default_value)

        model_learning_rate = param_dict.get("model_learning_rate", config_space["model_learning_rate"].default_value)
        model_weight_decay = param_dict.get("model_weight_decay", config_space["model_weight_decay"].default_value)
        num_cem_iters = int(param_dict.get("num_cem_iters", config_space["num_cem_iters"].default_value))
        cem_popsize = int(param_dict.get("cem_popsize", config_space["cem_popsize"].default_value))
        cem_alpha = param_dict.get("cem_alpha", config_space["cem_alpha"].default_value)
        num_cem_elites = int(param_dict.get("cem_elites_ratio", config_space["cem_elites_ratio"].default_value) * cem_popsize)
        model_train_epoch = int(param_dict.get("model_train_epoch", config_space["model_train_epoch"].default_value))
        plan_hor = int(param_dict.get("plan_hor", config_space["plan_hor"].default_value))


        # Instantiation of parameter vector
        overrides = [
            ("ctrl_cfg.opt_cfg.cfg.max_iters", num_cem_iters),
            ("ctrl_cfg.opt_cfg.cfg.popsize", cem_popsize),
            ("ctrl_cfg.opt_cfg.cfg.num_elites", num_cem_elites),
            ("ctrl_cfg.opt_cfg.cfg.alpha", cem_alpha),
            ("ctrl_cfg.opt_cfg.plan_hor", plan_hor),
            ("ctrl_cfg.prop_cfg.model_train_cfg.epochs", model_train_epoch)
        ]

        cfg, cfg_module = cfg_creator(overrides)

        def nn_constructor(model_init_cfg):
            model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
                name="model",
                num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
                sess=cfg_module.SESS, load_model=model_init_cfg.get("load_model", False),
                model_dir=model_init_cfg.get("model_dir", None)
            ))
            if not model_init_cfg.get("load_model", False):
                model.add(FC(
                    hidden_layer_width, input_dim=cfg_module.MODEL_IN,
                    activation=self.activation_fns[act_idx], weight_decay=model_weight_decay
                ))
                for _ in range(num_hidden_layers - 1):
                    model.add(FC(
                        hidden_layer_width, activation=self.activation_fns[act_idx], weight_decay=model_weight_decay
                    ))
                model.add(FC(cfg_module.MODEL_OUT, weight_decay=model_weight_decay))
            model.finalize(self.model_optimizers[model_opt_idx], {"learning_rate": model_learning_rate})
            return model

        cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_constructor = nn_constructor

        # Build up model
        policy = MPC(cfg.ctrl_cfg)
        return policy

def main(args):
    exp = PETS_Trainer(args)
    # run experiment
    exp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-ctrl_type', type=str, default='MPC',
                        help='Control type will be applied (default: MPC)')
    parser.add_argument('-config_id', type=str, default='0',
                        help="Configuration id for logging")
    parser.add_argument('-seed', type=int, default=0,
                        help='Random seed of the experiment')
    parser.add_argument('-budget', type=int, default=301,
                        help='Number of trials of the experiments')
    parser.add_argument('-last_k', type=int, default=3,
                        help='How many steps of returns will be averaged to get the final score')
    parser.add_argument('-num_hidden_layers', type=int, default=4,
                        help='Number of hidden layers of the architecture')
    parser.add_argument('-hidden_layer_width', type=int, default=200,
                        help='Number of hidden units per layer of the architecture')
    parser.add_argument('-act_idx', type=str, default='swish',
                        help='Which activation function to use, [swish, sigmoid, tanh, relu]')
    parser.add_argument('-schedule_name', type=str, default='random_static_cem_opt',
                        help='See ds/Schedule.py file')
    args = parser.parse_args()
    print(args)
    main(args) 