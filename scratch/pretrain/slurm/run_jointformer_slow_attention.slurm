#!/bin/bash

#SBATCH --job-name=jointformer-slow-attention                             
#SBATCH --output jointformer-slow-attention.log                              
#SBATCH --error jointformer-slow-attention.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_long
                                
#SBATCH --cpus-per-task=8                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=96:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=100                                                             


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

srun python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer/results/pre-train/slow_attention \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_dataset_config configs/datasets/guacamol/physchem \
    --path_to_tokenizer_config configs/tokenizers/smiles_separate_task_token \
    --path_to_model_config configs/models/jointformer_no_flash_attention \
    --path_to_trainer_config configs/trainers/joint \
    --path_to_logger_config configs/loggers/wandb \
    