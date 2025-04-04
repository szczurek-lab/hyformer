#!/bin/bash

python3 experiments/data/subset_data.py \
        --seed 0 \
        --num_samples 10000 \
        --data_path /home/adamizdebski/files/data/datasets/guacamol/train/smiles.txt \
        --output /home/adamizdebski/files/data/datasets/guacamol/train/10000/seed_0/smiles.txt \
