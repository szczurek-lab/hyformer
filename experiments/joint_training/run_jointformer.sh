#!/bin/bash

python3 experiments/joint_training/train.py \
    --data_dir /home/adamizdebski/files/data \
    --out_dir /home/adamizdebski/files/results/jointformer/smoke_test \
    --path_to_dataset_config configs/datasets/guacamol \
    --path_to_tokenizer_config configs/tokenizers/smiles \
    --path_to_model_config configs/models/jointformer_test \
    --path_to_trainer_config configs/trainers/test \
    --path_to_logger_config configs/loggers/wandb_test \
    