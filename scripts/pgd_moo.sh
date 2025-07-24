#!/bin/bash

# Synthetic Functions 
# "dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 dtlz7 zdt1 zdt2 zdt3 zdt4 zdt6"

# RE
# "re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61"

seeds="0 1 2 3 4"
tasks="dtlz1 dtlz2 dtlz3 dtlz4 zdt1 zdt2 zdt3 zdt4 zdt6 dtlz5 dtlz6 dtlz7 re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61"
model="Diffusion-Guidance"
train_modes="Crowding SubCrowding"

for task in $tasks; do 
    for train_mode in $train_modes; do
        echo "Running $model on $task and train mode $train_mode"
        python off_moo_baselines/diffusion_guidance/experiment.py --model=$model --task=$task --use_wandb=False --retrain_model=False --train_mode=$train_mode --seed=$seed & 
    done
done 


wait