#!/bin/bash

set -e

python ./experiments/constrained_optimization/task.py \
    --out_dir /raid/aizd/hybrid_transformer/results/constrained_optimization_v1/ \
    --objective qed logp binding \
    --backbone Jointformer \
    --method bayesian_posterior \
    --similarity_constraint 0.4 0.6 \
    --reference_file_dir /raid/aizd/data/guacamol_v1_all.smiles \
    --debug TRUE \
