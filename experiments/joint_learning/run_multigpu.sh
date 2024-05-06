#!/bin/bash

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=gpu \
    experiments/joint_learning/train_multigpu.py \
    --out_dir /raid/aizd/jointformer/results_ablation/joint_learning/guacamol \
    --path_to_task_config configs/tasks/guacamol/qed \
    --path_to_model_config configs/models/jointformer \
    --path_to_trainer_config configs/trainers/pretrain \
