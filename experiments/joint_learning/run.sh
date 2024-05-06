#!/bin/bash

python experiments/joint_learning/train_multigpu.py \
    --out_dir /raid/aizd/jointformer/results/test \
    --path_to_task_config configs/tasks/guacamol/qed \
    --path_to_model_config configs/models/jointformer \
    --path_to_trainer_config configs/trainers/new \
    --dev_mode True
