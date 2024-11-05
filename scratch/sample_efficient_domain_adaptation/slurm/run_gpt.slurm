#!/bin/bash

#SBATCH --job-name=finetune-gpt                           
#SBATCH --output logs/logs/%x_%j.log                              
#SBATCH --error logs/logs/%x_%j.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
                                
#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=24:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=100                                        


source $HOME/.bashrc

conda deactivate
conda activate jointformer

model_name="gpt"
model_seed=1337
fractions_train_dataset=(1.0)
dataset_names=("lipo" "esol" "freesolv")
metric="rmse"

data_dir="/lustre/groups/aih/jointformer/data"
root_dir="/lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation"
path_to_tokenizer_config="configs/tokenizers/gpt_tokenizer"
path_to_model_config="configs/models/${model_name}_prediction"
path_to_trainer_config="configs/trainers/sample_efficient_domain_adaptation"

for dataset_name in "${dataset_names[@]}"; do
   for fraction_train_dataset in "${fractions_train_dataset[@]}"; do
      echo "Running python script for $dataset_name dataset with fraction_train_examples = $fraction_train_dataset."
      
      python "experiments/data_efficient_domain_adaptation/run.py" \
         --out_dir $root_dir/$model_name/molecule_net/scaffold/$dataset_name/$fraction_train_dataset/$dataset_seed \
         --data_dir $data_dir \
         --path_to_dataset_config configs/datasets/molecule_net/scaffold/$dataset_name \
         --path_to_tokenizer_config configs/tokenizers/${model_name}_tokenizer \
         --path_to_model_config configs/models/${model_name}_prediction \
         --path_to_trainer_config configs/trainers/finetune \
         --fraction_train_dataset $fraction_train_dataset \
         --path_to_model_ckpt /lustre/groups/aih/jointformer/results/pretrain/gpt/seed_1337/ckpt.pt \
         --model_seed $model_seed \
         --hyperparameters_grid_filepath 'experiments/data_efficient_domain_adaptation/hyperparameters_grid.json' \
         --optuna_metric_direction 'minimize' \
         --optuna_n_trials 50 \
         --optuna_n_jobs 12 \
         --num_seeds 4 \
         --metric $metric
   done

   echo "Aggregating results for $dataset_name dataset."
   python "experiments/data_efficient_domain_adaptation/aggregate_results.py" \
      --out_dir $root_dir/$model_name/molecule_net/scaffold/$dataset_name \
      --metric $metric

done

echo "Script finished."
