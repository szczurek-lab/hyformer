#!/bin/bash

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=gpu \
    experiments/joint_training/train.py \
    --out_dir results/test \
    --path_to_task_config configs/tasks/moses/unsupervised \
    --path_to_model_config configs/models/jointformer_tiny \
    --path_to_trainer_config configs/trainers/joint \
    --path_to_logger_config configs/loggers/wandb
