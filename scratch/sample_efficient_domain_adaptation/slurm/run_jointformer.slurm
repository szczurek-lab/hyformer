#!/bin/bash

#SBATCH --job-name=finetune-jointformer                           
#SBATCH --output logs/%x/%j.log                              
#SBATCH --error logs/%x/%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal

#SBATCH --array=0-11    
#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=48:00:00                                    

#SBATCH --nice=100                                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer

### Define constants

model_name="jointformer"
dataset_names=("lipo" "esol" "freesolv")
objectives=("generation", "generation_physchem", "generation_mlm_physchem", "generation_mlm")
num_runs=1


### Define job array specific variables

tasks=()
for dataset_name in "${dataset_names[@]}"; do
  for objective in "${objectives[@]}"; do
      tasks+=("$dataset_name,$objective")
  done
done

task=${tasks[$SLURM_ARRAY_TASK_ID]}

dataset_name=$(echo $task | cut -d',' -f1)
objective=$(echo $task | cut -d',' -f2)

echo dataset_name: $dataset_name
echo objective: $objective


### Run the experiment

for run in $(seq 1 $num_runs); do
   
   # set directories
   root_dir="/lustre/groups/aih/jointformer"
   out_dir=$root_dir/results/finetune/$model_name/$objective/molecule_net/scaffold/$dataset_name/seed_$run
   data_dir=$root_dir/"data"
   
   echo run: $root_dir/results/pretrain/$model_name/$objective/seed_1337/ckpt.pt

   python "experiments/data_efficient_domain_adaptation/run.py" \
      --out_dir $out_dir \
      --data_dir $data_dir \
      --path_to_dataset_config configs/datasets/molecule_net/scaffold/$dataset_name \
      --path_to_model_config configs/models/jointformer_v2_prediction \
      --path_to_tokenizer_config configs/tokenizers/smiles_separate_task_token \
      --path_to_trainer_config configs/trainers/finetune \
      --path_to_model_ckpt $root_dir/results/pretrain/$model_name/pretrain_${objective}/seed_1337/ckpt.pt \
      --hyperparameters_grid_filepath 'experiments/data_efficient_domain_adaptation/hyperparameters_grid.json' \
      --optuna_metric_direction 'minimize' \
      --optuna_n_trials 50 \
      --optuna_n_jobs 1 \
      --fraction_train_dataset 1.0 \
      --model_seed 1337 \
      --seed $run \
      --metric "rmse"
done

echo "Script finished."
