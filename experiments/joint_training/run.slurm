#!/bin/bash

#SBATCH --job-name=jointformer-pretrain                              
#SBATCH --output output-%x.txt                              
#SBATCH --error error-%x.txt                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_reservation
#SBATCH --reservation=reservations_supergpu07

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                 
#SBATCH --gpus-per-task=1                                   
#SBATCH --cpus-per-task=8                                   
#SBATCH --mem=32G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=24:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=10000                                        
#SBATCH --constraint=gpu_tesla_A100                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer/results_pretrain \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_task_config configs/tasks/guacamol/physchem \
    --path_to_model_config configs/models/jointformer \
    --path_to_trainer_config configs/trainers/joint \
    --path_to_logger_config configs/loggers/wandb 
