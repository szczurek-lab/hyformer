#!/bin/bash

model_seeds=(1337)
dataset_seeds=(0)
fractions_train_dataset=(1.0)
dataset_names=("freesolv")
python_test_script="experiments/data_efficient_domain_adaptation/test.py"
results_aggregation_script="experiments/data_efficient_domain_adaptation/aggregate_results.py"

for dataset_name in "${dataset_names[@]}"; do
   for fraction_train_dataset in "${fractions_train_dataset[@]}"; do
      for dataset_seed in "${dataset_seeds[@]}"; do
         for model_seed in "${model_seeds[@]}"; do
            echo "Running python script for $dataset_name dataset with fraction_train_examples = $fraction_train_dataset, seed $dataset_seed, model seed $model_seed."
            
            python "experiments/data_efficient_domain_adaptation/train.py" \
               --data_dir /home/adamizdebski/files/jointformer/data \
               --out_dir /home/adamizdebski/files/results/data_efficient_domain_adaptation/jointformer/molecule_net/$dataset_name/$fraction_train_dataset/$dataset_seed \
               --path_to_dataset_config configs/datasets/molecule_net/$dataset_name \
               --path_to_tokenizer_config configs/tokenizers/smiles \
               --path_to_model_config configs/models/jointformer \
               --path_to_trainer_config configs/trainers/finetune \
               --fraction_train_dataset $fraction_train_dataset \
               --dataset_seed $dataset_seed \
               --model_seed $model_seed \
               --path_to_model_ckpt '/home/adamizdebski/files/results/jointformer/ckpt.pt'

            python "experiments/data_efficient_domain_adaptation/test.py" \
               --data_dir /home/adamizdebski/files/jointformer/data \
               --out_dir /home/adamizdebski/files/results/data_efficient_domain_adaptation/jointformer/molecule_net/$dataset_name/$fraction_train_dataset/$dataset_seed \
               --path_to_dataset_config configs/datasets/molecule_net/$dataset_name \
               --path_to_tokenizer_config configs/tokenizers/smiles \
               --path_to_model_config configs/models/jointformer \
               --path_to_trainer_config configs/trainers/finetune \
               --model_seed $model_seed \
               --metric rmse

         done
      done
   done

   echo "Aggregating results for $dataset_name dataset."
   
   python "experiments/data_efficient_domain_adaptation/aggregate_results.py" \
      --out_dir /home/adamizdebski/files/results/data_efficient_domain_adaptation/jointformer/molecule_net/$dataset_name \
      --metric rmse

done

echo "Script finished."
