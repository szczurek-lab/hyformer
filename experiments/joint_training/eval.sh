#!/bin/bash

python experiments/joint_training/eval.py \
    --benchmark guacamol \
    --out_dir results/test \
    --generated_file_path results/test/generated/smiles.txt \
    --reference_file_path data/moses/train/smiles.txt \
