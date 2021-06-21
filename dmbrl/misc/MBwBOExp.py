from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from config_space.config_space import DEFAULT_CONFIGSPACE

import os
from time import sleep
import pickle

import tensorflow as tf
from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent
from bohb.HPWorker import HPWorker
from hpbandster.optimizers import BOHB, RandomSearch, HyperBand
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import ConfigSpace.read_and_write.json as pcs_out
from dmbrl.modeling.utils.HPO import build_policy_constructor

from tqdm import tqdm
import json

class MBWithBOExperiment:
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
                .bo_cfg:
                    .min_budget (int): Minimal budget to run one configuration.
                        Default to 4.
                    .max_budget (int): Maximal budget to run one configuration.
                        Default to 15.
                    .nopt_iter (int): Number of configurations to be evaluated.
                        Default to 4
                    .eta (int): Base number of Hyperband
                    .n_workers (int): Number of workers to run at the same time.
                        Default to 2
            args:

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
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")
        self.cfg_creator = cfg_creator

        self.logdir = get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory.")
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        self.nfinal_eval = params.log_cfg.get("nfinal_eval", 0)
        # Hyperparameters for BOHB
        self.min_budget = params.bo_cfg.get("min_budget", 10)
        self.max_budget = params.bo_cfg.get("max_budget", 100)
        self.nopt_iter = params.bo_cfg.get("nopt_iter", 100)
        self.eta = params.bo_cfg.get("eta", 3)
        self.last_k = params.bo_cfg.get("last_k", 1)

        self.optimizer_type = args.opt_type
        self.start_worker = args.worker
        self.interface = args.interface
        self.config_names = args.config_names
        self.run_id = args.run_id
        self.shared_directory = os.path.join(self.logdir, "share_file")

        self.worker_id = args.worker_id
        # self.log_worker_dir = os.path.join(self.logdir, 'worker_'+str(self.worker_id))
        self.env_name = args.env

        self.control_args = {key: val for (key, val) in args.ctrl_arg}
        self.prop_type = self.control_args['prop-type']
        self.opt_type = self.control_args['opt-type']
        self.translate_into_model = build_policy_constructor(self)

    def run_experiment(self):
        """Perform experiment.
        """
        # os.makedirs(self.log_worker_dir, exist_ok=True)
        os.makedirs(self.shared_directory, exist_ok=True)
        host = hpns.nic_name_to_host(self.interface)
        # Create worker
        if self.start_worker:
            sleep(20)   # short artificial delay to make sure the nameserver is already running
            worker = HPWorker(run_id=self.run_id,
                              host=host,
                              env=self.env_name,
                              config_names=self.config_names,
                              train_func=self.train_func,
                              last_k=self.last_k)
            print("Created Worker")
            worker.load_nameserver_credentials(working_directory=self.shared_directory)
            print("Loaded Server")
            worker.run(background=False)
            exit(0)
        
        cs = HPWorker.get_configspace(self.env_name, self.config_names)
        # Write configuration space
        with open(os.path.join(self.shared_directory, 'configspace.json'), "w") as f:
            f.write(pcs_out.write(cs))

        # Init Result logger
        result_logger = hpres.json_result_logger(directory=self.shared_directory, overwrite=True)
        # Start server
        NS = hpns.NameServer(run_id=self.run_id, host=host, port=0, working_directory=self.shared_directory)
        ns_host, ns_port = NS.start()
        
        worker = HPWorker(run_id=self.run_id,
                          host=host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          env=self.env_name,
                          config_names=self.config_names,
                          train_func=self.train_func,
                          last_k =self.last_k)
        worker.run(background=True)
        # Random optimizer
        if self.optimizer_type == "random":
            opt = RandomSearch(configspace=cs,
                        run_id=self.run_id,
                        host=host,
                        nameserver=ns_host,
                        nameserver_port=ns_port,
                        eta=self.eta,
                        result_logger=result_logger,
                        min_budget=self.min_budget,
                        max_budget=self.max_budget,
                        )
        elif self.optimizer_type == "bohb":
            # BOHB optimizer
            opt = BOHB(configspace=cs,
                        run_id=self.run_id,
                        host=host,
                        nameserver=ns_host,
                        nameserver_port=ns_port,
                        eta=self.eta,
                        result_logger=result_logger,
                        min_budget=self.min_budget,
                        max_budget=self.max_budget
                        )
        elif self.optimizer_type == "hyperband":
            # BOHB optimizer
            opt = HyperBand(configspace=cs,
                    run_id=self.run_id,
                    host=host,
                    nameserver=ns_host,
                    nameserver_port=ns_port,
                    eta=self.eta,
                    result_logger=result_logger,
                    min_budget=self.min_budget,
                    max_budget=self.max_budget
                )
        else:
            raise NotImplementedError("Not Implemented hyperparameter optimizer")

        res = opt.run(n_iterations=self.nopt_iter, min_n_workers=1)
        
        result_file = os.path.join(self.shared_directory, 'bohb_result.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump(res, f)

        # Shutdown the server
        opt.shutdown(shutdown_workers=True)
        NS.shutdown()


    def train_func(self, config_id, config, budget):
        tqdm.write("####################################################################")
        tqdm.write(json.dumps(config))
        policy = self.translate_into_model(config, DEFAULT_CONFIGSPACE)

        # For training
        traj_obs, traj_acs, traj_rets, traj_rews, traj_init_states = [], [], [], [], []
        # For evaluation
        traj_eval_obs, traj_eval_acs, traj_eval_rews, traj_eval_rets, traj_eval_init_states = [], [], [], [], []

        traj_train_losses, traj_test_losses, traj_episode_losses = [], [], []

        
        config_directory = os.path.join(self.logdir, "%d_%d_%d" %config_id)
        # Perform initial rollouts
        for i in range(self.ninit_rollouts):
            samples = []
            samples.append(
                self.agent.sample(
                    self.task_hor, policy
                )
            )
            traj_obs.extend([sample["obs"] for sample in samples])
            traj_acs.extend([sample["ac"] for sample in samples])
            traj_rews.extend([sample["rewards"] for sample in samples])
            traj_init_states.extend([sample["init_state"] for sample in samples])
            traj_train_loss, traj_test_loss, episode_loss_lst  = policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
            )
            traj_train_losses.append(traj_train_loss)
            traj_test_losses.append(traj_test_loss)
            traj_episode_losses.append(episode_loss_lst)
        # Training loop
        for i in range(int(budget)):
            # Change to current hyperparameters
            tqdm.write("Starting training iteration (budget) %d/%d." % (i + 1, budget))
            iter_dir = os.path.join(config_directory, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            # Rollout nrecord episodes
            for j in range(self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, policy,
                        os.path.join(iter_dir, "rollout%d.mp4" % j)
                    )
                )
            if self.nrecord > 0:
                for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
                    os.remove(os.path.join(iter_dir, item))
            
            for j in range(max(self.neval, self.nrollouts_per_iter) - self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, policy
                    )
                )
            
            # Append training data
            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.append([sample["reward_sum"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            traj_init_states.extend([sample["init_state"] for sample in samples[:self.nrollouts_per_iter]])
            # Append evaluating data
            traj_eval_obs.extend([sample["obs"] for sample in samples[self.nrollouts_per_iter:self.neval]])
            traj_eval_acs.extend([sample["ac"] for sample in samples[self.nrollouts_per_iter:self.neval]])
            traj_eval_rets.append([sample["reward_sum"] for sample in samples[self.nrollouts_per_iter:self.neval]])
            traj_eval_rews.extend([sample["rewards"] for sample in samples[self.nrollouts_per_iter:self.neval]])
            traj_eval_init_states.extend([sample["init_state"] for sample in samples[self.nrollouts_per_iter:self.neval]])
            tqdm.write("Rewards obtained: %s" %json.dumps([sample["reward_sum"] for sample in samples]))
            samples = samples[:self.nrollouts_per_iter]

            policy.dump_logs(config_directory, iter_dir)
            # Delete iteration directory if not used
            try:
                if os.path.exists(iter_dir):
                    if len(os.listdir(iter_dir)) == 0:
                        try:
                            os.rmdir(iter_dir)
                        except FileNotFoundError:
                            pass
            except PermissionError:
                pass
            
            if i < budget - 1:
                traj_train_loss, traj_test_loss, episode_loss_lst  = policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
                traj_train_losses.append(traj_train_loss)
                traj_test_losses.append(traj_test_loss)
                traj_episode_losses.append(episode_loss_lst)
            savemat(
                os.path.join(config_directory, "logs.mat"),
                {
                    "train_observations": traj_obs,
                    "train_actions": traj_acs,
                    "train_returns": traj_rets,
                    "train_rewards": traj_rews,
                    "train_init_state": traj_init_states,
                    "eval_observations": traj_eval_obs,
                    "eval_actions": traj_eval_acs,
                    "eval_returns": traj_eval_rets,
                    "eval_rewards": traj_eval_rews,
                    "eval_init_state": traj_eval_init_states,
                    "traj_train_loss": traj_train_losses,
                    "traj_test_loss": traj_test_losses,
                    "traj_episode_losses": traj_episode_losses,  
                }
            )
            policy.save_model(config_directory)


        # Final Evaluation
        if self.nfinal_eval > 0:
            samples = []
            for k in range(self.nfinal_eval):
                samples.append(
                    self.agent.sample(
                        self.task_hor, policy
                    )
                )
            tqdm.write("Final Rewards obtained: %s" %json.dumps([sample["reward_sum"] for sample in samples]))
            savemat(
                os.path.join(config_directory, "logs.mat"),
                {
                    "train_observations": traj_obs,
                    "train_actions": traj_acs,
                    "train_returns": traj_rets,
                    "train_rewards": traj_rews,
                    "train_init_state": traj_init_states,
                    "eval_observations": traj_eval_obs,
                    "eval_actions": traj_eval_acs,
                    "eval_returns": traj_eval_rets,
                    "eval_rewards": traj_eval_rews,
                    "eval_init_state": traj_eval_init_states,
                    "traj_train_loss": traj_train_losses,
                    "traj_test_loss": traj_test_losses,
                    "final_eval_returns": [sample["reward_sum"] for sample in samples]
                }
            )
            policy.save_model(config_directory)           
        
        return traj_rets, traj_eval_rets
