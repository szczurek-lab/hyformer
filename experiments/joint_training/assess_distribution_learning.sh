model=$1
ckpt=$2

python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt ${ckpt} \
        --path_to_model_config configs/models/${model} \
        --path_to_tokenizer_config configs/tokenizers/smiles \
        --chembl_training_file data/guacamol/train/smiles.txt \
        --output outputs/${model}/assess_distribution_learning.js
