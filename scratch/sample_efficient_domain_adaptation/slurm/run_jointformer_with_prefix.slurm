#!/bin/bash

#SBATCH --job-name=sample-efficient-domain-adaptation-jointformer                             
#SBATCH --output sample-efficient-domain-adaptation-jointformer.log                              
#SBATCH --error sample-efficient-domain-adaptation-jointformer.err                                 

#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
                                
#SBATCH --cpus-per-task=12                                   
#SBATCH --mem=20G                                           
#SBATCH --gres=gpu:1                                        
#SBATCH --time=2:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=100                                       


source $HOME/.bashrc

conda deactivate
conda activate jointformer-experiments

model_seeds=(1337)
dataset_seeds=(0 1 2)
fractions_train_dataset=(0.01 0.1 1.0)
dataset_names=("lipo")
jointformer="jointformer_with_prefix_big"
tokenizer="smiles_with_prefix"

for dataset_name in "${dataset_names[@]}"; do
   for fraction_train_dataset in "${fractions_train_dataset[@]}"; do
      for dataset_seed in "${dataset_seeds[@]}"; do

         for model_seed in "${model_seeds[@]}"; do

            echo "Running python script for $dataset_name dataset with fraction_train_examples = $fraction_train_dataset, seed $dataset_seed, model seed $model_seed."
            
            python "experiments/data_efficient_domain_adaptation/train.py" \
               --data_dir /lustre/groups/aih/jointformer/data \
               --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/$jointformer/molecule_net/scaffold/$dataset_name/$fraction_train_dataset/$dataset_seed \
               --path_to_dataset_config configs/datasets/molecule_net/scaffold/$dataset_name \
               --path_to_tokenizer_config configs/tokenizers/$tokenizer \
               --path_to_model_config configs/models/$jointformer \
               --path_to_trainer_config configs/trainers/sample_efficient_domain_adaptation \
               --fraction_train_dataset $fraction_train_dataset \
               --dataset_seed $dataset_seed \
               --model_seed $model_seed \
               --path_to_model_ckpt /lustre/groups/aih/jointformer/results/pre-train/3lm_psychem_long_big/prefix_token/cls_emb/bs_512/drop_0/seed_1337/ckpt.pt \

            python "experiments/data_efficient_domain_adaptation/test.py" \
               --data_dir /lustre/groups/aih/jointformer/data \
               --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/$jointformer/molecule_net/scaffold/$dataset_name/$fraction_train_dataset/$dataset_seed \
               --path_to_dataset_config configs/datasets/molecule_net/scaffold/$dataset_name \
               --path_to_tokenizer_config configs/tokenizers/$tokenizer \
               --path_to_model_config configs/models/$jointformer \
               --path_to_trainer_config configs/trainers/sample_efficient_domain_adaptation \
               --model_seed $model_seed \
               --destroy_ckpt \
               --metric rmse

         done
      done
   done

   echo "Aggregating results for $dataset_name dataset."
   
   python "experiments/data_efficient_domain_adaptation/aggregate_results.py" \
      --out_dir /lustre/groups/aih/jointformer/results/sample_efficient_domain_adaptation/$jointformer/molecule_net/scaffold/$dataset_name \
      --metric rmse

done

echo "Script finished."
