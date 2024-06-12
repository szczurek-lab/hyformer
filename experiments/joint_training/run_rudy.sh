#!/bin/bash

python experiments/joint_training/train.py \
    --data_dir /raid/aizd/jointformer/data \
    --out_dir /raid/aizd/jointformer/results/test \
    --path_to_task_config configs/tasks/guacamol/debug \
    --path_to_model_config configs/models/jointformer_debug \
    --path_to_trainer_config configs/trainers/joint \
    --path_to_logger_config configs/loggers/wandb \
    #    --logger_display_name jointformer \
