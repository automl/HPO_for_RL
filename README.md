# Author:
Baohe Zhang

# Code Instruction

### Overview:
This code is mainly based on Kurtland Chua's implementation of PETS paper. See [code](https://github.com/kchua/handful-of-trials).

Besides that, to inject the bayesian optimization and Populaion-based Training optimization methods, several components are added. 

config_space (folder):
**config_space.py** : Define the configuration space of hyperparameters for each environment. It is also the key component.

pbt (folder):
This folder contains the codes for Population-based training methods and Population-based training with back-tracking. The experiment object can be found pbt-mbexp.py and pbt-bt-mbexp.py. 

bohb (folder):
This folder contains the bayesian optimization method [BOHB](https://github.com/automl/HpBandSter/blob/master/hpbandster/optimizers/bohb.py). This also provides Hyperband and Random search functionalities.

### Environment Setting
You can use anaconda environment yml file if you want which contains all the necessary libraries used to run this code.

### Scripts for Running codes
In order to run the experiments given a fixed hyperparameters, you can refer the following scripts. In the scripts folder, there are some examples when using slurm cluster.

Example commend to start the code:
```
# xxx refers the cluster partation
# Hyperband
cd scripts
sbatch -p xxx -a 1-10 hyperband.sh
# For PBT

# PBT
cd scripts
sbatch -p xxx -a 1-10 pbt.sh

# PBT-BT
cd scripts
sbatch -p xxx -a 1-10 pbt-bt.sh

```

### Learned Schedules
In the learned_schedule folder, we have uploaded few hyperparameter schedules that are learned by running HPO methods on PETS. You should be able to reproduce the results reported in our [paper](https://arxiv.org/abs/2102.13651). The evaluation function is in eval_schedules.py file.

### Disclaimer
This code is still not yet perfectly cleaned up and might contain issues. Feel free to contact me if there are anything unclear to you.

### License
This code follows the MIT License.

### Known Issue
When running PBT on Slurm-based cluster, the controller will occupy one gpu without really using it since the controller will not train any model. One possible way to alleviate this temporarly is by starting the controller first and then start the workers. But this would need the user taking care that the workers know the directory which stores the server connect file.