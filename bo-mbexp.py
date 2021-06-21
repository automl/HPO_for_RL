from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint

from dotmap import DotMap

from dmbrl.misc.MBwBOExp import MBWithBOExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config

import tensorflow as tf
import numpy as np
import random

def create_cfg_creator(env, ctrl_type, ctrl_args, base_overrides, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    def cfg_creator(additional_overrides=None):
        if additional_overrides is not None:
            return create_config(env, ctrl_type, ctrl_args, base_overrides + additional_overrides, logdir)
        return create_config(env, ctrl_type, ctrl_args, base_overrides, logdir)

    return cfg_creator


def main(args):
    cfg_creator = create_cfg_creator(args.env, args.ctrl_type, args.ctrl_arg, args.override, args.logdir)
    cfg = cfg_creator()[0]
    cfg.pprint()

    if args.ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    exp = MBWithBOExperiment(cfg.exp_cfg, cfg_creator, args)
    os.makedirs(exp.logdir, exist_ok=True)
    
    exp.run_experiment()


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
    # Parser for running BOHB on cluster 
    parser.add_argument('-worker', action='store_true',
                        help='Flag to turn this into a worker process')
    parser.add_argument('-interface', type=str, default='lo',
                        help='Interface name to use for creating host')
    parser.add_argument('-run_id', type=str, default='111',
                        help='A unique run id for this optimization run. An easy option is to use'
                             ' the job id of the clusters scheduler.')
    parser.add_argument('-worker_id', type=int, default=0,
                        help='The ID of the worker')
    parser.add_argument('-config_names', type=str, default="model_learning_rate", nargs="+",
                        help='Specify which hyperparameters to optimize)')
    parser.add_argument('-opt_type', type=str, default="bohb",
                        help='Specify which optimizer to use')
    parser.add_argument('-seed', type=int, default=0,
                        help='Specify the random seed to use')
    args = parser.parse_args()
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(args)
    main(args) 
