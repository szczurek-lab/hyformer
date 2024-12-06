#!/bin/bash

#SBATCH --job-name=finetune_gpt                          
#SBATCH --output logs/%x/%j.log                              
#SBATCH --error logs/%x/%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --array=0-2                                
#SBATCH --cpus-per-task=20                                 
#SBATCH --mem=80G                                           
#SBATCH --gres=gpu:2                                        
#SBATCH --time=48:00:00                                    

#SBATCH --nice=100                                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer

ROOT_DIR="/lustre/groups/aih/jointformer/results"
DATA_DIR="/lustre/groups/aih/jointformer/data"
DATASET_NAMES=("lipo" "esol" "freesolv")

tasks=()
for dataset_name in "${DATASET_NAMES[@]}"; do
      tasks+=("$dataset_name")
done

task=${tasks[$SLURM_ARRAY_TASK_ID]}
DATASET_NAME=$(echo $task | cut -d',' -f1)
STUDY_NAME="gpt_${DATASET_NAME}"
    
    
python experiments/molecule_net/optuna_distributed_worker.py \
    --out-dir "${ROOT_DIR}/optuna_hpo_final/molecule_net/gpt/${DATASET_NAME}" \
    --data_dir $DATA_DIR \
    --optuna-n-jobs 1 \
    --sampler "grid" \
    --search_space_filepath "experiments/molecule_net/optuna_hparams_search_space.json" \
    --hparams_grid_filepath "experiments/molecule_net/optuna_hparams_grid.json" \
    --path_to_dataset_config configs/datasets/molecule_net/scaffold/$DATASET_NAME \
    --path_to_model_config configs/models/gpt_prediction \
    --path_to_tokenizer_config configs/tokenizers/gpt_tokenizer \
    --path_to_trainer_config configs/trainers/finetune \
    --path_to_model_ckpt $ROOT_DIR/pretrain/gpt/seed_1337/ckpt.pt \
    