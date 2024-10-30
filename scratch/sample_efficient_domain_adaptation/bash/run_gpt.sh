#!/bin/bash


model_name="gpt"
model_seed=1337
fractions_train_dataset=(0.1)
dataset_names=("lipo")
metric="rmse"


for dataset_name in "${dataset_names[@]}"; do
   for fraction_train_dataset in "${fractions_train_dataset[@]}"; do
      echo "Running python script for $dataset_name dataset with fraction_train_examples = $fraction_train_dataset."

      python3 "experiments/data_efficient_domain_adaptation/run.py" \
         --out_dir /home/adamizdebski/files/results/jointformer/data_efficient_domain_adaptation/$model_name/molecule_net/$dataset_name/$fraction_train_dataset \
         --data_dir /home/adamizdebski/files/jointformer/data \
         --path_to_dataset_config configs/datasets/molecule_net/scaffold/$dataset_name \
         --path_to_tokenizer_config configs/tokenizers/${model_name}_tokenizer \
         --path_to_model_config configs/models/${model_name}_prediction \
         --path_to_trainer_config configs/trainers/finetune \
         --fraction_train_dataset $fraction_train_dataset \
         --model_seed $model_seed \
         --hyperparameters_grid_filepath 'experiments/data_efficient_domain_adaptation/hyperparameters_grid.json' \
         --optuna_metric_direction 'minimize' \
         --optuna_n_trials 2 \
         --optuna_n_jobs 2 \
         --num_seeds 2 \
         --metric $metric \
         --destroy_ckpt \
         --test \
         # --path_to_model_ckpt '/home/adamizdebski/files/results/jointformer/ckpt.pt' \

   done

   echo "Aggregating results for $dataset_name dataset."
   python "experiments/data_efficient_domain_adaptation/aggregate_results.py" \
      --out_dir /home/adamizdebski/files/results/jointformer/data_efficient_domain_adaptation/$model_name/molecule_net/$dataset_name \
      --metric $metric

done

echo "Script finished."
