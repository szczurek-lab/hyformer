#!/bin/bash

#SBATCH --job-name=torch-test                               # create a short name for your job
#SBATCH --output output-%x.txt                              # create an output file
#SBATCH --error error-%x.txt                                # create an error file 

#SBATCH -p gpu_p
#SBATCH --qos gpu_priority                                  # gpu_priority for debugging, gpu_long for training 

#SBATCH --nodes=1                                           # node count
#SBATCH --ntasks=1                                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                                   # cpu-cores per task (>1 if multi-threaded tasks) // GPU queue node cores
#SBATCH --mem=4G                                            # total memory per node // GPU queue node memory
#SBATCH --gres=gpu:1                                        # number of gpus per node
#SBATCH --time=00:05:00                                     # total run time limit (HH:MM:SS)

#SBATCH --mail-type=begin                                   # send mail when job begins
#SBATCH --mail-type=end                                     # send mail when job ends
#SBATCH --mail-type=fail                                    # send mail if job fails
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=10000                                        # priority (niceness) // GPU queue priority
#SBATCH --constraint=gpu_tesla_A100                         # GPU type


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

echo "Running on $(hostname)"
echo "-------------------------------------------------"
echo -n "Number of available GPUs: "
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
echo -n "GPU name: "
nvidia-smi --query-gpu=name --format=csv,noheader
echo -n "The highest compatible CUDA version: "
nvcc --version | grep release | awk '{print $5}'
echo "-------------------------------------------------"

echo "Starting the PyTorch compatibility test script..."
python experiments/joint_training/test.py
