#!/bin/bash

#SBATCH --job-name=chemberta-domain-efficient                              
#SBATCH --output output-%x.log                              
#SBATCH --error error-%x.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                 
#SBATCH --gpus-per-task=1                                   
#SBATCH --cpus-per-task=8                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=8:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=999                                       
#SBATCH --constraint=gpu_tesla_A100                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

# srun python experiments/data_efficient_domain_adaptation/run.py \
#    --data_dir /lustre/groups/aih/jointformer/data \
#    --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/chemberta/esol \
#    --path_to_dataset_config configs/datasets/molecule_net/esol \
#    --path_to_tokenizer_config configs/tokenizers/chemberta \
#    --path_to_model_config configs/models/chemberta \
#    --path_to_trainer_config configs/trainers/finetune \
#    --path_to_logger_config configs/loggers/wandb \

# srun python experiments/data_efficient_domain_adaptation/run.py \
#    --data_dir /lustre/groups/aih/jointformer/data \
#    --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/chemberta/freesolv \
#    --path_to_dataset_config configs/datasets/molecule_net/freesolv \
#    --path_to_tokenizer_config configs/tokenizers/chemberta \
#    --path_to_model_config configs/models/chemberta \
#    --path_to_trainer_config configs/trainers/finetune \
#    --path_to_logger_config configs/loggers/wandb \

srun python experiments/data_efficient_domain_adaptation/run.py \
   --data_dir /lustre/groups/aih/jointformer/data \
   --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation_select_best/chemberta/lipo_scaffold \
   --path_to_dataset_config configs/datasets/molecule_net/lipo_scaffold \
   --path_to_tokenizer_config configs/tokenizers/chemberta \
   --path_to_model_config configs/models/chemberta \
   --path_to_trainer_config configs/trainers/finetune \
   --path_to_logger_config configs/loggers/wandb \
 