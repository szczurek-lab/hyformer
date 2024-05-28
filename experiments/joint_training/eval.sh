#!/bin/bash

python experiments/joint_training/eval.py \
    --benchmark moses \
    --out_dir results/test \
    --generated_file_path results/test/generated/smiles.txt \
