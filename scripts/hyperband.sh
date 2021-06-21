#!/bin/bash

# Example command

# activate your conda env
source ~/.bashrc
source activate mbrl

INTERFACE=`ip route get 8.8.8.8 | cut -d' ' -f5 | head -1`
echo Interface read:$INTERFACE

# ENV=halfcheetah_v3
# TASK_HORIZON=1000
ENV=reacher
TASK_HORIZON=10

RUN_ID=$SLURM_ARRAY_JOB_ID
# Replace with your directory
DIR=log/$ENV\_$RUN_ID
NINIT_ROLLOUTS=1
MIN_BUDGET=79
MAX_BUDGET=80
NOPT_ITER=5
LAST_K=3
ETA=2
NEVAL=1

SEED=0
OPT_TYPE=hyperband # or [random, bohb]
PROP_TYPE=TSinf

OPT_TYPE_PETS=CEM
# CONFIG_NAMES=plan_hor\ num_cem_iters\ cem_popsize\ cem_elites_ratio\ cem_alpha
CONFIG_NAMES=model_weight_decay\ model_learning_rate\ model_train_epoch

cd ..

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
then
   python -u bo-mbexp.py -config_names $CONFIG_NAMES \
   -opt_type $OPT_TYPE \
   -run_id $RUN_ID \
   -env $ENV \
   -logdir $DIR \
   -worker_id $SLURM_ARRAY_TASK_ID \
   -seed $SEED \
   -interface $INTERFACE \
   -o exp_cfg.log_cfg.neval $NEVAL \
   -o exp_cfg.exp_cfg.ninit_rollouts $NINIT_ROLLOUTS \
   -o exp_cfg.bo_cfg.min_budget $MIN_BUDGET \
   -o exp_cfg.bo_cfg.max_budget $MAX_BUDGET \
   -o exp_cfg.bo_cfg.nopt_iter $NOPT_ITER \
   -o exp_cfg.bo_cfg.eta $ETA \
   -o exp_cfg.bo_cfg.last_k $LAST_K \
   -o exp_cfg.sim_cfg.task_hor $TASK_HORIZON \
   -ca prop-type $PROP_TYPE \
   -ca opt-type $OPT_TYPE_PETS
else
   python -u bo-mbexp.py -config_names $CONFIG_NAMES \
   -opt_type $OPT_TYPE \
   -run_id $RUN_ID \
   -worker \
   -env $ENV \
   -logdir $DIR \
   -worker_id $SLURM_ARRAY_TASK_ID \
   -seed $SEED \
   -interface $INTERFACE \
   -o exp_cfg.log_cfg.neval $NEVAL \
   -o exp_cfg.exp_cfg.ninit_rollouts $NINIT_ROLLOUTS \
   -o exp_cfg.bo_cfg.min_budget $MIN_BUDGET \
   -o exp_cfg.bo_cfg.max_budget $MAX_BUDGET \
   -o exp_cfg.bo_cfg.nopt_iter $NOPT_ITER \
   -o exp_cfg.bo_cfg.eta $ETA \
   -o exp_cfg.bo_cfg.last_k $LAST_K \
   -o exp_cfg.sim_cfg.task_hor $TASK_HORIZON \
   -ca prop-type $PROP_TYPE \
   -ca opt-type $OPT_TYPE_PETS
fi

