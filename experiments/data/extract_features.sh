python3 experiments/data/extract_features.py \
        --path_to_model_ckpt /home/jano1906/git/jointformer/checkpoints/GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl \
        --path_to_model_config configs/models/moler \
        --path_to_task_config configs/tasks/guacamol/physchem/ \
        --split test \
        --output outputs/moler/test/guacamol_v1_features.npy \
        --chunk_size 30000


#python3 experiments/data/extract_features.py \
#        --path_to_model_ckpt /home/jano1906/git/jointformer/checkpoints/ckpt.pt \
#        --path_to_model_config /home/jano1906/git/jointformer/configs/models/jointformer \
#        --path_to_task_config configs/tasks/guacamol/physchem/ \
#        --split train \
#        --output outputs/jointformer/train/guacamol_v1_features.npy

#python3 experiments/data/extract_features.py \
#        --path_to_model_ckpt /home/jano1906/git/jointformer/checkpoints/ckpt.pt \
#        --path_to_model_config /home/jano1906/git/jointformer/configs/models/jointformer \
#        --path_to_task_config configs/tasks/guacamol/physchem/ \
#        --split val \
#        --output outputs/jointformer/val/guacamol_v1_features.npy
#

#python3 experiments/data/extract_features.py \
#        --path_to_model_ckpt /home/jano1906/git/jointformer/checkpoints/ckpt.pt \
#        --path_to_model_config /home/jano1906/git/jointformer/configs/models/jointformer \
#        --path_to_task_config configs/tasks/guacamol/physchem/ \
#        --split test \
#        --output outputs/jointformer/test/guacamol_v1_features.npy