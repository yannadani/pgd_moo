tasks="dtlz1 dtlz2 dtlz3 dtlz4 dtlz5 dtlz6 dtlz7"
model="MultipleModels"
train_modes="Vallina COM IOM RoMA ICT TriMentoring"

for task in $tasks; do 
    for train_mode in $train_modes; do
        echo "Running $model on $task and train mode $train_mode"
        python off_moo_baselines/multi_head/experiment.py --model=$model --task=$task --use_wandb=False --retrain_model=False --train_mode=$train_mode --seed=$seed & 
    done
done 


wait