#!/bin/bash

#SBATCH --job-name=jointformer-pretrain-with-prefix-big                             
#SBATCH --output jointformer-pretrain-with-prefix.log                              
#SBATCH --error jointformer-pretrain-with-prefix.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_long
                                
#SBATCH --cpus-per-task=16                                   
#SBATCH --mem=80G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=96:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=1000                                        
#SBATCH --reservation=res_adam.izdebski                          


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

srun python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer/results/pre-train/3lm_psychem_long_big/prefix_token/cls_emb/bs_512/drop_0/ \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_dataset_config configs/datasets/guacamol/physchem \
    --path_to_tokenizer_config configs/tokenizers/smiles_with_prefix \
    --path_to_model_config configs/models/jointformer_with_prefix_big \
    --path_to_trainer_config configs/trainers/joint_long \
    --path_to_logger_config configs/loggers/wandb \
