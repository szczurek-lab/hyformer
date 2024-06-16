#!/bin/bash

#SBATCH --job-name=jointformer-ddp                        # create a short name for your job
#SBATCH --output output-%x.txt                            # create an output file
#SBATCH --error error-%x.txt                              # create an error file 

#SBATCH -p gpu_p
#SBATCH --qos gpu_reservation                             # gpu_priority for debugging, gpu_long for training 
#SBATCH --reservation=reservations_supergpu07

#SBATCH --nodes=1                                         # node count
#SBATCH --ntasks-per-node=8                               # total number of tasks across all nodes
#SBATCH --gpus-per-task=1                                 # gpus per task
#SBATCH --cpus-per-task=8                                 # cpu-cores per task (>1 if multi-threaded tasks) // GPU queue node cores
#SBATCH --mem=32G                                         # total memory per node // GPU queue node memory
#SBATCH --gres=gpu:8                                      # number of gpus per node
#SBATCH --time=24:00:00                                   # total run time limit (HH:MM:SS)

#SBATCH --mail-type=begin                                   # send mail when job begins
#SBATCH --mail-type=end                                     # send mail when job ends
#SBATCH --mail-type=fail                                    # send mail if job fails
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=10000                                        # priority (niceness) // GPU queue priority
#SBATCH --constraint=gpu_tesla_A100                         # GPU type


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments


srun python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=gpu \
    experiments/joint_training/train.py \
        --out_dir /lustre/groups/aih/jointformer/results \
        --data_dir /lustre/groups/aih/jointformer/data \
        --logger_display_name jointformer-ddp \
        --path_to_task_config configs/tasks/guacamol/debug \
        --path_to_model_config configs/models/jointformer \
        --path_to_trainer_config configs/trainers/debug \
        --path_to_logger_config configs/loggers/wandb \
