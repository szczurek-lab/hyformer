#!/bin/bash

#python ./scripts/joint_learning/train_old.py \
#    --out_dir /raid/aizd/jointformer/results_ablation/joint_learning/guacamol \
#    --benchmark guacamol \
#    --model HybridTransformer \
#    --task_p 0.9

#python ./scripts/joint_learning/eval.py \
#    --out_dir /raid/aizd/jointformer/results_ablation/joint_learning/guacamol \
#    --benchmark guacamol \
#    --model HybridTransformer \
#    --task_p 0.9

python ./scripts/joint_learning/train.py \
    --out_dir /raid/aizd/hybrid_transformer/results_ablation/joint_learning/guacamol \
    --benchmark guacamol \
    --model HybridTransformer \
    --task_p 0.15

python ./scripts/joint_learning/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results_ablation/joint_learning/guacamol \
    --benchmark guacamol \
    --model HybridTransformer \
    --task_p 0.15

#python ./scripts/joint_learning/train_old.py \
#    --out_dir /raid/aizd/jointformer/results_ablation/joint_learning/guacamol \
#    --benchmark guacamol \
#    --model HybridTransformer \
#    --task_p 1.

#python ./scripts/joint_learning/eval.py \
#    --out_dir /raid/aizd/jointformer/results_ablation/joint_learning/guacamol \
#    --benchmark guacamol \
#    --model HybridTransformer \
#    --task_p 1.
