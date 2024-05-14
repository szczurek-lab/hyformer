#!/bin/bash

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=gpu \
    experiments/joint_learning/train.py \
    --out_dir results \
    --path_to_task_config configs/tasks/guacamol/physchem \
    --path_to_model_config configs/models/jointformer \
    --path_to_trainer_config configs/trainers/new \
    --path_to_logger_config configs/loggers/wandb \
