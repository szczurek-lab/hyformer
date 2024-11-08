#!/bin/bash

#SBATCH --job-name=evaluate-gpt                           
#SBATCH --output logs/%x/%j.log                              
#SBATCH --error logs/%x/%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
                                
#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=02:00:00                                    

#SBATCH --nice=100                                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer

root_dir="/lustre/groups/aih/jointformer"
python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt ${root_dir}/results/pretrain/gpt/seed_1337/ckpt.pt \
        --path_to_model_config configs/models/gpt \
        --path_to_tokenizer_config configs/tokenizers/gpt_tokenizer \
        --chembl_training_file $root_dir/data/data/guacamol/train/smiles.txt \
        --output $root_dir/results/access_distribution_learning/$model_name/guacamol/assess_distribution_learning.js
