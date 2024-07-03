#!/bin/bash

#SBATCH --job-name=test-ddp                              
#SBATCH --output output-%x.log                              
#SBATCH --error error-%x.log                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_priority

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=2                                 
#SBATCH --gpus-per-task=1                                   
#SBATCH --cpus-per-task=8                                   
#SBATCH --mem=16G                                           
#SBATCH --gres=gpu:2                                        
#SBATCH --time=00:30:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=10000                                        
#SBATCH --constraint=gpu_tesla_A100                        

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

srun python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer/results_test \
    --data_dir /lustre/groups/aih/jointformer/data \
    --path_to_task_config configs/tasks/guacamol/physchem \
    --path_to_model_config configs/models/jointformer_test \
    --path_to_trainer_config configs/trainers/test \
    --path_to_logger_config configs/loggers/wandb_test
