tasks="dtlz5 dtlz6 dtlz7"
model="MultiHead"
train_modes="Vallina GradNorm PcGrad"

for task in $tasks; do 
    for train_mode in $train_modes; do
        echo "Running $model on $task and train mode $train_mode"
        python off_moo_baselines/multi_head/experiment.py --model=$model --task=$task --use_wandb=False --retrain_model=False --train_mode=$train_mode --seed=$seed & 
    done
done 


wait