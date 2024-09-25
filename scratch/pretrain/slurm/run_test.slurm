#!/bin/bash

#SBATCH --job-name=pretrain-test                         
#SBATCH --output logs/%x_%j.log                              
#SBATCH --error logs/%x_%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_priority
                                
#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=00:30:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=100                                            


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

srun python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer/results/pretrain/test \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_dataset_config configs/datasets/guacamol/physchem \
    --path_to_tokenizer_config configs/tokenizers/smiles \
    --path_to_model_config configs/models/jointformer_test \
    --path_to_trainer_config configs/trainers/test
    