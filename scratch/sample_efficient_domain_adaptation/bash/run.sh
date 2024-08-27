#!/bin/bash

python experiments/data_efficient_domain_adaptation/run.py \
   --data_dir /lustre/groups/aih/jointformer/data \
   --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/freesolv \
   --path_to_dataset_config configs/datasets/molecule_net/freesolv \
   --path_to_tokenizer_config configs/tokenizers/smiles \
   --path_to_model_config configs/models/jointformer_test \
   --path_to_trainer_config configs/trainers/finetune \
   --path_to_model_ckpt /lustre/groups/aih/jointformer/results/lm_psychem/no_separate_token/shared_emb/bs_512/drop_2/seed_1337/ckpt.pt \
   --dry_run \
   --prepare_data \
