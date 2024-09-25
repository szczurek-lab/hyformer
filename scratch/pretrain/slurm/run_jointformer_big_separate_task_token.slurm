#!/bin/bash

#SBATCH --job-name=jointformer-big-separate-task-token                              
#SBATCH --output jointformer-big-separate-task-token.log                              
#SBATCH --error jointformer-big-separate-task-token.err                                

#SBATCH -p gpu_p
#SBATCH --qos gpu_long

#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=40G                                           
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
    --out_dir /lustre/groups/aih/jointformer/results/pre-train/jointformer-big-task-token-95-gen \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_dataset_config configs/datasets/guacamol/physchem \
    --path_to_tokenizer_config configs/tokenizers/smiles_separate_task_token \
    --path_to_model_config configs/models/jointformer_big_separate_task_token \
    --path_to_trainer_config configs/trainers/joint_95 \
    --path_to_logger_config configs/loggers/wandb \
    