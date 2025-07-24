#!/bin/bash

seeds="0 1 2 3 4"
tasks="re23 re24 re25"
model="ParetoFlow"
train_modes="Vallina"

for task in $tasks; do 
    for train_mode in $train_modes; do
        echo "Running $model on $task and train mode $train_mode"
        python off_moo_baselines/pareto_flow/experiment.py --model=$model --task=$task --use_wandb=False --retrain_model=False --train_mode=$train_mode --seed=$seed & 
    done
done 


wait