#!/bin/bash

# activate your conda env
source ~/.bashrc
source activate mbrl

SAMPLE_FROM_PERCENT=0.2
RESAMPLE_IF_NOT_IN_PERCENT=0.8
RESAMPLE_PROBABILITY=0.25


POPULATION_SIZE=40
CRITERION_MODE=lastk
TASK_HORIZON=10
LAST_K=3
INITIAL_STEP=6
BUDGET=60
STEP=5

PROP_TYPE=E
OPT_TYPE_PETS=CEM
# TOTAL STEPS = (BUDGET - 1) * STEP + INITIAL_STEP

ENV=halfcheetah_v3

# Replace with your directory
DIR=log/$ENV\_$SLURM_ARRAY_JOB_ID
WORKER_ID=$SLURM_ARRAY_JOB_ID

# CONFIG_NAMES=plan_hor\ num_cem_iters\ cem_popsize\ cem_elites_ratio\ cem_alpha
CONFIG_NAMES=model_weight_decay\ model_learning_rate\ model_train_epoch
NEVAL=1

cd ..

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
then
   python -u pbt-mbexp.py -config_names $CONFIG_NAMES \
   -seed 0 \
   -env $ENV \
   -logdir $DIR \
   -worker_id $SLURM_ARRAY_TASK_ID \
   -sample_from_percent $SAMPLE_FROM_PERCENT \
   -resample_if_not_in_percent $RESAMPLE_IF_NOT_IN_PERCENT \
   -resample_probability $RESAMPLE_PROBABILITY \
   -o exp_cfg.log_cfg.neval $NEVAL \
   -o exp_cfg.pbt_cfg.pop_size $POPULATION_SIZE \
   -o exp_cfg.pbt_cfg.budget $BUDGET \
   -o exp_cfg.pbt_cfg.criterion_mode $CRITERION_MODE \
   -o exp_cfg.pbt_cfg.last_k $LAST_K \
   -o exp_cfg.pbt_cfg.initial_step $INITIAL_STEP \
   -o exp_cfg.pbt_cfg.step $STEP \
   -o exp_cfg.sim_cfg.task_hor $TASK_HORIZON \
   -ca prop-type $PROP_TYPE \
   -ca opt-type $OPT_TYPE_PETS
else
   python -u pbt-mbexp.py -config_names $CONFIG_NAMES \
   -seed 0 \
   -worker \
   -env $ENV \
   -logdir $DIR \
   -worker_id $WORKER_ID \
   -o exp_cfg.log_cfg.neval $NEVAL \
   -o exp_cfg.pbt_cfg.pop_size $POPULATION_SIZE \
   -o exp_cfg.pbt_cfg.budget $BUDGET \
   -o exp_cfg.pbt_cfg.criterion_mode $CRITERION_MODE \
   -o exp_cfg.pbt_cfg.last_k $LAST_K \
   -o exp_cfg.pbt_cfg.initial_step $INITIAL_STEP \
   -o exp_cfg.pbt_cfg.step $STEP \
   -o exp_cfg.sim_cfg.task_hor $TASK_HORIZON \
   -ca prop-type $PROP_TYPE \
   -ca opt-type $OPT_TYPE_PETS
fi