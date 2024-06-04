#!/bin/bash

#SBATCH --job-name=jointformer-train                        # create a short name for your job
#SBATCH --output output-%x.txt                              # create an output file
#SBATCH --error error-%x.txt                                # create an error file 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal                                      # gpu_priority for debugging, gpu_long for training 

#SBATCH --nodes=1                                           # node count
#SBATCH --ntasks=1                                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                                   # cpu-cores per task (>1 if multi-threaded tasks) // GPU queue node cores
#SBATCH --mem=4G                                            # total memory per node // GPU queue node memory
#SBATCH --gres=gpu:1                                        # number of gpus per node
#SBATCH --time=24:00:00                                     # total run time limit (HH:MM:SS)

#SBATCH --mail-type=begin                                   # send mail when job begins
#SBATCH --mail-type=end                                     # send mail when job ends
#SBATCH --mail-type=fail                                    # send mail if job fails
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=10000                                        # priority (niceness) // GPU queue priority
#SBATCH --constraint=gpu_tesla_A100                         # GPU type


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

python experiments/joint_training/train.py \
    --out_dir /lustre/groups/aih/jointformer \
    --logger_display_name jointformer \
    --path_to_task_config configs/tasks/guacamol/physchem \
    --path_to_model_config configs/models/jointformer \
    --path_to_trainer_config configs/trainers/joint \
    --path_to_logger_config configs/loggers/wandb \
    --dev_mode False
