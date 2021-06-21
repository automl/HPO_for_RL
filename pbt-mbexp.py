from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse

from dotmap import DotMap

from dmbrl.misc.MBwPBTExp import MBWithPBTExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config

import tensorflow as tf
import numpy as np

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

    exp = MBWithPBTExperiment(cfg.exp_cfg, cfg_creator, args)    
    exp.run_experiment()


if __name__ == "__main__":
    print("Exp start!")
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah, halfcheetah_v3]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-ctrl_type', type=str, default='MPC',
                        help='Control type will be applied (default: MPC)')                
    # Parser for running dynamic scheduler on cluster 
    parser.add_argument('-config_names', type=str, default="model_learning_rate", nargs="+",
                        help='Specify which hyperparameters to optimize')
    parser.add_argument('-seed', type=int, default=0,
                        help='Specify the random seed of the experiment')
    parser.add_argument('-worker_id', type=int, default=0,
                        help='The worker id, e.g. using SLURM ARRAY JOB ID')
    parser.add_argument('-worker', action='store_true',
                        help='Flag to turn this into a worker process otherwise this will start a new controller')
    parser.add_argument('-sample_from_percent', type=float, default=0.2,
                        help='Sample from the top ratio N')
    parser.add_argument('-resample_if_not_in_percent', type=float, default=0.8,
                        help='Resample if the configuration is not in the top ratio N')
    parser.add_argument('-resample_probability', type=float, default=0.25,
                        help='Probability of an exploited member resampling configurations randomly')
    parser.add_argument('-resample_prob_decay', type=float, default=1,
                        help='decay factor of resample if not in percent')
    parser.add_argument('-not_copy_data', type=bool, default=False,
                        help='Set to True if not copy the data to new trials')
    
    args = parser.parse_args()
    print(args)
    # Set the random seeds of the experiment
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    main(args) 
