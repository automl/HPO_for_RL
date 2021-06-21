import tensorflow as tf
from dmbrl.controllers import MPC
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
from dotmap import DotMap


def build_policy_constructor(exp):
    """Returns a function which will translates a parameter vector into a configuration
    that can be used to construct a policy.

    Args:
        exp: experiment object

    Returns:
        translate_into_model: A function which will construct a controller (e.g. MPC) given the hyperparameters
    """    
    def translate_into_model(param_dict, config_space, reset=True):
        """Translate a set of hyperparameters into a policy

        Args:
            param_dict: dictonary which contains the hyperparameters of the policy
            config_space: Configuration space which will be used to determine the default hyperparameter
            reset (bool, optional): If True, reset the default graph of Tensorflow. Defaults to True.

        Returns:
            policy: MPC controller
        """        
        # Clear the graph to avoid conflicts with previous graphs.
        if reset:
            tf.reset_default_graph()

        # Give parameters names to avoid bugs
        config_dict = dict()
        config_space = config_space[exp.env_name]
        for config_name in config_space.keys():
            if config_name in exp.config_names:
                config_dict[config_name] = param_dict[config_name]
            else:
                config_dict[config_name] = config_space[config_name].default_value 

        num_hidden_layers = int(config_dict["num_hidden_layers"])
        hidden_layer_width = int(config_dict["hidden_layer_width"])
        act_idx = config_dict["act_idx"]
        model_learning_rate = config_dict["model_learning_rate"]
        model_weight_decay = config_dict["model_weight_decay"]
        model_opt_idx = config_dict["model_opt_idx"]
        num_cem_iters = int(config_dict["num_cem_iters"])
        cem_popsize = int(config_dict["cem_popsize"])
        cem_alpha = config_dict["cem_alpha"]
        num_cem_elites = int(config_dict["cem_elites_ratio"]*cem_popsize)
        model_train_epoch = int(config_dict["model_train_epoch"])
        plan_hor = int(config_dict["plan_hor"])


        # Instantiation of parameter vector

        if exp.opt_type == 'CEM':
            overrides = [
                ("ctrl_cfg.opt_cfg.cfg.max_iters", num_cem_iters),
                ("ctrl_cfg.opt_cfg.cfg.popsize", cem_popsize),
                ("ctrl_cfg.opt_cfg.cfg.num_elites", num_cem_elites),
                ("ctrl_cfg.opt_cfg.cfg.alpha", cem_alpha),
                ("ctrl_cfg.opt_cfg.plan_hor", plan_hor),
                ("ctrl_cfg.prop_cfg.model_train_cfg.epochs", model_train_epoch)
            ]
        elif exp.opt_type == 'Random':
            overrides = [
                ("ctrl_cfg.opt_cfg.cfg.popsize", cem_popsize),
                ("ctrl_cfg.opt_cfg.plan_hor", plan_hor),
                ("ctrl_cfg.prop_cfg.model_train_cfg.epochs", model_train_epoch)
            ]
        else:
            print(exp.opt_type)
            raise(NotImplementedError("Given control type %s is unknown" %exp.opt_type))

        cfg, cfg_module = exp.cfg_creator(overrides)

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
                    activation=exp.activation_fns[act_idx], weight_decay=model_weight_decay
                ))
                for _ in range(num_hidden_layers - 1):
                    model.add(FC(
                        hidden_layer_width, activation=exp.activation_fns[act_idx], weight_decay=model_weight_decay
                    ))
                model.add(FC(cfg_module.MODEL_OUT, weight_decay=model_weight_decay))
            model.finalize(exp.model_optimizers[model_opt_idx], {"learning_rate": model_learning_rate})
            return model

        cfg.ctrl_cfg.prop_cfg.model_init_cfg.model_constructor = nn_constructor

        # Build up model
        policy = MPC(cfg.ctrl_cfg)

        return policy

    return translate_into_model