#!/bin/bash

#SBATCH --job-name=jointformer-old-pretrain                              
#SBATCH --output output-%x.log                              
#SBATCH --error error-%x.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                 
#SBATCH --gpus-per-task=1                                   
#SBATCH --cpus-per-task=8                                   
#SBATCH --mem=16G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=36:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=10000                                        
#SBATCH --constraint=gpu_tesla_A100                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

srun python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer/results/3lm_mlm/no_separate_token/shared_emb/bs_512/drop_2/ \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_task_config configs/tasks/guacamol/physchem \
    --path_to_model_config configs/models/jointformer \
    --path_to_trainer_config configs/trainers/joint_old \
    --path_to_logger_config configs/loggers/wandb \
    --path_to_model_ckpt /lustre/groups/aih/jointformer/results/3lm_mlm/no_separate_token/shared_emb/bs_512/drop_2/seed_1337/ckpt.pt \
