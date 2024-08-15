model=$1
ckpt=$2

python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt ${ckpt} \
        --path_to_model_config configs/models/${model} \
        --path_to_tokenizer_config configs/tokenizers/smiles \
        --chembl_training_file data/guacamol/train/smiles.txt \
        --output outputs/${model}/assess_distribution_learning.js

#python3 experiments/joint_training/assess_distribution_learning.py \
#        --path_to_model_ckpt /home/jano1906/git/jointformer/checkpoints/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl \
#        --path_to_model_config configs/models/moler \
#        --path_to_task_config configs/tasks/guacamol/physchem/ \
#        --chembl_training_file ~/datasets/guacamol/guacamol_v1_train.smiles \
#        --output outputs/output.js