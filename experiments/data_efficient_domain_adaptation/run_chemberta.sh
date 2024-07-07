#!/bin/bash

python experiments/data_efficient_domain_adaptation/run.py \
   --data_dir /home/adamizdebski/files/jointformer/data \
   --out_dir /home/adamizdebski/files/chemberta/results/data_efficient_domain_adaptation \
   --path_to_dataset_config configs/datasets/molecule_net/freesolv \
   --path_to_tokenizer_config configs/tokenizers/chemberta \
   --path_to_model_config configs/models/chemberta \
   --path_to_trainer_config configs/trainers/finetune \
   --dry_run
