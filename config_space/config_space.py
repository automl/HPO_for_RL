import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

"""
This defines the search space of each environment. The default value is chosen based on Kurtland et al. 2018.
See https://github.com/kchua/handful-of-trials for more information
"""
DEFAULT_CONFIGSPACE = {
    "cartpole": {
        # Model Architecture
        "num_hidden_layers" : CSH.UniformIntegerHyperparameter("num_hidden_layers", lower=2, upper=8, default_value=3, log=False),
        "hidden_layer_width" : CSH.UniformIntegerHyperparameter("hidden_layer_width", lower=100, upper=600, default_value=500, log=True),
        "act_idx" : CSH.CategoricalHyperparameter(name='activation', choices=['relu', 'tanh', 'sigmoid', 'swish'], default_value='swish'),
        # Model Optimizer
        "model_learning_rate" : CSH.UniformFloatHyperparameter('model_learning_rate', lower=1e-5, upper=4e-2, default_value=1e-3, log=True),
        "model_weight_decay" : CSH.UniformFloatHyperparameter('model_weight_decay', lower=1e-7, upper=1e-1, default_value=0.00025, log=True),
        "model_opt_idx" : CSH.CategoricalHyperparameter(name="model_opt_idx", choices=['adam', 'adadelta', 'adagrad', 'sgd', 'rms'], default_value='adam'),
        "model_train_epoch" : CSH.UniformIntegerHyperparameter("model_train_epoch", lower=3, upper=15, default_value=5, log=False),
        # Planner
        "num_cem_iters" : CSH.UniformIntegerHyperparameter("num_cem_iters", lower=4, upper=6, default_value=5, log=False),
        "cem_popsize" : CSH.UniformIntegerHyperparameter("cem_popsize", lower=200, upper=700, default_value=400, log=True),
        "cem_alpha" : CSH.UniformFloatHyperparameter("cem_alpha", lower=0.05, upper=0.2, default_value=0.1, log=False),
        "cem_elites_ratio" : CSH.UniformFloatHyperparameter("cem_elites_ratio", lower=0.04, upper=0.5, default_value=0.1, log=True),
        "plan_hor" : CSH.UniformIntegerHyperparameter("plan_hor", lower=5, upper=40, default_value=25, log=False),
    },
    "pusher": {
        # Model Architecture
        "num_hidden_layers" : CSH.UniformIntegerHyperparameter("num_hidden_layers", lower=3, upper=8, default_value=3, log=False),
        "hidden_layer_width" : CSH.UniformIntegerHyperparameter("hidden_layer_width", lower=100, upper=600, default_value=200, log=True),
        "act_idx" : CSH.CategoricalHyperparameter(name='activation', choices=['relu', 'tanh', 'sigmoid', 'swish'], default_value='swish'),
        # Model Optimizer
        "model_learning_rate" : CSH.UniformFloatHyperparameter('model_learning_rate', lower=3e-5, upper=3e-3, default_value=0.001, log=True),
        "model_weight_decay" : CSH.UniformFloatHyperparameter('model_weight_decay', lower=1e-7, upper=1e-1, default_value=5e-4, log=True),
        "model_opt_idx" : CSH.CategoricalHyperparameter(name="model_opt_idx", choices=['adam', 'adadelta', 'adagrad', 'sgd', 'rms'], default_value='adam'),
        "model_train_epoch" : CSH.UniformIntegerHyperparameter("model_train_epoch", lower=3, upper=20, default_value=5, log=False),
        # Planners 
        "num_cem_iters" : CSH.UniformIntegerHyperparameter("num_cem_iters", lower=3, upper=10, default_value=5, log=False),
        "cem_popsize" : CSH.UniformIntegerHyperparameter("cem_popsize", lower=100, upper=700, default_value=500, log=True),
        "cem_alpha" : CSH.UniformFloatHyperparameter("cem_alpha", lower=0.01, upper=0.5, default_value=0.1, log=False),
        "cem_elites_ratio" : CSH.UniformFloatHyperparameter("cem_elites_ratio", lower=0.04, upper=0.5, default_value=0.1, log=True),
        "plan_hor" : CSH.UniformIntegerHyperparameter("plan_hor", lower=5, upper=40, default_value=25, log=False),
    },
    "reacher": {
        # Model Architecture
        "num_hidden_layers" : CSH.UniformIntegerHyperparameter("num_hidden_layers", lower=3, upper=8, default_value=4, log=False),
        "hidden_layer_width" : CSH.UniformIntegerHyperparameter("hidden_layer_width", lower=100, upper=600, default_value=200, log=True),
        "act_idx" : CSH.CategoricalHyperparameter(name='activation', choices=['relu', 'tanh', 'sigmoid', 'swish'], default_value='swish'),
        # Model Optimizer
        "model_learning_rate" : CSH.UniformFloatHyperparameter('model_learning_rate', lower=1e-5, upper=4e-2, default_value=0.00075, log=True),
        "model_weight_decay" : CSH.UniformFloatHyperparameter('model_weight_decay', lower=1e-7, upper=1e-1, default_value=0.0005, log=True),
        "model_opt_idx" : CSH.CategoricalHyperparameter(name="model_opt_idx", choices=['adam', 'adadelta', 'adagrad', 'sgd', 'rms'], default_value='adam'),
        "model_train_epoch" : CSH.UniformIntegerHyperparameter("model_train_epoch", lower=3, upper=20, default_value=5, log=False),
        # Planner
        "num_cem_iters" : CSH.UniformIntegerHyperparameter("num_cem_iters", lower=4, upper=6, default_value=5, log=False),
        "cem_popsize" : CSH.UniformIntegerHyperparameter("cem_popsize", lower=200, upper=700, default_value=400, log=True),
        "cem_alpha" : CSH.UniformFloatHyperparameter("cem_alpha", lower=0.05, upper=0.4, default_value=0.1, log=False),
        "cem_elites_ratio" : CSH.UniformFloatHyperparameter("cem_elites_ratio", lower=0.04, upper=0.5, default_value=0.1, log=True),
        "plan_hor" : CSH.UniformIntegerHyperparameter("plan_hor", lower=5, upper=40, default_value=25, log=False),
    },
    "halfcheetah_v3": {
        # Model Architecture
        "num_hidden_layers" : CSH.UniformIntegerHyperparameter("num_hidden_layers", lower=2, upper=8, default_value=4, log=False),
        "hidden_layer_width" : CSH.UniformIntegerHyperparameter("hidden_layer_width", lower=100, upper=600, default_value=200, log=False),
        "act_idx" : CSH.CategoricalHyperparameter(name='activation', choices=['relu', 'tanh', 'sigmoid', 'swish'], default_value='swish'),
        # Model Optimizer
        "model_learning_rate" : CSH.UniformFloatHyperparameter('model_learning_rate', lower=1e-5, upper=4e-2, default_value=0.001, log=True),
        "model_weight_decay" : CSH.UniformFloatHyperparameter('model_weight_decay', lower=1e-7, upper=1e-1, default_value=0.000075, log=True),
        "model_opt_idx" : CSH.CategoricalHyperparameter(name="model_opt_idx", choices=['adam', 'adadelta', 'adagrad', 'sgd', 'rms'], default_value='adam'),
        "model_train_epoch" : CSH.UniformIntegerHyperparameter("model_train_epoch", lower=3, upper=20, default_value=5, log=False),
        # Planner
        "num_cem_iters" : CSH.UniformIntegerHyperparameter("num_cem_iters", lower=3, upper=8, default_value=5, log=False),
        "cem_popsize" : CSH.UniformIntegerHyperparameter("cem_popsize", lower=200, upper=700, default_value=500, log=True),
        "cem_alpha" : CSH.UniformFloatHyperparameter("cem_alpha", lower=0.05, upper=0.2, default_value=0.1, log=False),
        "cem_elites_ratio" : CSH.UniformFloatHyperparameter("cem_elites_ratio", lower=0.04, upper=0.5, default_value=0.1, log=True),
        "plan_hor" : CSH.UniformIntegerHyperparameter("plan_hor", lower=5, upper=60, default_value=30, log=False),
    },
    "hopper": {
        # Model Architecture
        "num_hidden_layers" : CSH.UniformIntegerHyperparameter("num_hidden_layers", lower=2, upper=8, default_value=4, log=False),
        "hidden_layer_width" : CSH.UniformIntegerHyperparameter("hidden_layer_width", lower=100, upper=600, default_value=200, log=False),
        "act_idx" : CSH.CategoricalHyperparameter(name='activation', choices=['relu', 'tanh', 'sigmoid', 'swish'], default_value='swish'),
        # Model Optimizer
        "model_learning_rate" : CSH.UniformFloatHyperparameter('model_learning_rate', lower=1e-5, upper=4e-2, default_value=0.001, log=True),
        "model_weight_decay" : CSH.UniformFloatHyperparameter('model_weight_decay', lower=1e-7, upper=1e-1, default_value=0.000075, log=True),
        "model_opt_idx" : CSH.CategoricalHyperparameter(name="model_opt_idx", choices=['adam', 'adadelta', 'adagrad', 'sgd', 'rms'], default_value='adam'),
        "model_train_epoch" : CSH.UniformIntegerHyperparameter("model_train_epoch", lower=3, upper=20, default_value=5, log=False),
        # Planner
        "num_cem_iters" : CSH.UniformIntegerHyperparameter("num_cem_iters", lower=3, upper=8, default_value=5, log=False),
        "cem_popsize" : CSH.UniformIntegerHyperparameter("cem_popsize", lower=200, upper=700, default_value=500, log=True),
        "cem_alpha" : CSH.UniformFloatHyperparameter("cem_alpha", lower=0.05, upper=0.2, default_value=0.1, log=False),
        "cem_elites_ratio" : CSH.UniformFloatHyperparameter("cem_elites_ratio", lower=0.04, upper=0.5, default_value=0.1, log=True),
        "plan_hor" : CSH.UniformIntegerHyperparameter("plan_hor", lower=5, upper=60, default_value=30, log=False),
    }
}
