#!/bin/bash

python ./scripts/joint_learning/train.py \
    --out_dir /raid/aizd/hybrid_transformer/results_v9/joint_learning/molecule_net \
    --benchmark molecule_net \
    --model HybridTransformer \
    --task_p 0.15

python ./scripts/joint_learning/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results_v9/joint_learning/molecule_net \
    --benchmark molecule_net \
    --model HybridTransformer \
    --task_p 0.15
