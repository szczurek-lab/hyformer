#!/bin/bash

#SBATCH --job-name=finetune-gpt-lipo                           
#SBATCH --output logs/%x/%j.log                              
#SBATCH --error logs/%x/%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
                                
#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=24:00:00                                    

#SBATCH --nice=100                                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer

model_name="gpt"
dataset_names=("esol")
num_runs=1

for dataset_name in "${dataset_names[@]}"; do
   for run in $(seq 1 $num_runs); do
    
      # set directories
      root_dir="/lustre/groups/aih/jointformer"
      out_dir=$root_dir/results/finetune/$model_name/molecule_net/scaffold/$dataset_name/seed_$run
      data_dir=$root_dir/"data"
      
      python "experiments/data_efficient_domain_adaptation/run.py" \
         --out_dir $out_dir \
         --data_dir $data_dir \
         --path_to_dataset_config configs/datasets/molecule_net/scaffold/$dataset_name \
         --path_to_tokenizer_config configs/tokenizers/gpt_tokenizer \
         --path_to_model_config configs/models/gpt_prediction \
         --path_to_trainer_config configs/trainers/finetune \
         --path_to_model_ckpt /lustre/groups/aih/jointformer/results/pretrain/gpt/seed_1337/ckpt.pt \
         --hyperparameters_grid_filepath 'experiments/data_efficient_domain_adaptation/hyperparameters_grid.json' \
         --optuna_metric_direction 'minimize' \
         --optuna_n_trials 50 \
         --optuna_n_jobs 1 \
         --fraction_train_dataset 1.0 \
         --model_seed 1337 \
         --seed $run \
         --metric "rmse"
   done
done

echo "Script finished."
