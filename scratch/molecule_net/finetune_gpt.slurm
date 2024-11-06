#!/bin/bash

#SBATCH --job-name=gpt                           
#SBATCH --output logs/%x/%j.log                              
#SBATCH --error logs/%x/%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_priority

#SBATCH --array=0-1
                                
#SBATCH --cpus-per-task=2                                 
#SBATCH --mem=1G                                           
#SBATCH --gres=gpu:2                                        
#SBATCH --time=00:05:00                                    

#SBATCH --nice=100                                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer

ROOT_DIR="lustre/groups/aih/jointformer/results/optuna_test"
DATASET_NAMES=("lipo" "esol" "freesolv") # define datasets to iterate over
NUM_RUNS=4 # define the number of runs to average results over 

export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID


for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    for RUN in $(seq 1 $NUM_RUNS); do
        STUDY_NAME="${SLURM_JOB_NAME}_${DATASET_NAME}_run_${RUN}"
        ROOT_DIR="${ROOT_DIR}/molecule_net/${DATASET_NAME}/run_${RUN}"
        echo "Starting run $RUN with study name $STUDY_NAME"
        
        python experiments/molecule_net/optuna_distributed_worker.py \
            --root-dir $ROOT_DIR \
            --study-name $STUDY_NAME \
            --direction "minimize" \
            --optuna-n-jobs $SLURM_ARRAY_TASK_COUNT \
            --optuna-n-trials 10 \
            --seed $RUN
    done
done
    