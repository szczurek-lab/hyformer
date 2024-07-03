python3 experiments/joint_training/assess_distribution_learning.py \
        --path_to_model_ckpt checkpoints/ckpt.pt \
        --path_to_model_config configs/models/jointformer \
        --path_to_task_config configs/tasks/guacamol/physchem/ \
        --chembl_training_file ~/datasets/guacamol/guacamol_v1_train.smiles \
        --output outputs/output.js