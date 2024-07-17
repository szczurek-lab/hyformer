#python3 experiments/joint_training/assess_distribution_learning.py \
#        --path_to_model_ckpt checkpoints/ckpt.pt \
#        --path_to_model_config configs/models/jointformer \
#        --path_to_task_config configs/tasks/guacamol/physchem/ \
#        --chembl_training_file ~/datasets/guacamol/guacamol_v1_train.smiles \
#        --output outputs/output.js

python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt /home/jano1906/git/jointformer/checkpoints/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl \
        --path_to_model_config configs/models/moler \
        --path_to_task_config configs/tasks/guacamol/physchem/ \
        --chembl_training_file ~/datasets/guacamol/guacamol_v1_train.smiles \
        --output outputs/output.js