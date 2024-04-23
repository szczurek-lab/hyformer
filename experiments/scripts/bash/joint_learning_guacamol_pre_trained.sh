#!/bin/bash

python ./scripts/joint_learning/train.py \
    --out_dir /raid/aizd/hybrid_transformer/results_v3/joint_learning/guacamol \
    --benchmark guacamol \
    --model GPTPreTrained

python ./scripts/joint_learning/eval.py \
    --out_dir /raid/aizd/hybrid_transformer/results_v3/joint_learning/guacamol \
    --benchmark guacamol \
    --model GPTPreTrained
