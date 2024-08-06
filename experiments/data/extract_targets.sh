# #!/bin/bash

# splits=("train" "val" "test")
# targets=("plogp" "qed" "guacamol_mpo" "physchem")

# for split in "${splits[@]}"
# do
#   for target in "${targets[@]}"
#   do
#      if [ "$split" == "train" ] && [ "$target" == "guacamol_mpo" ]; then
#     continue # Skip the current iteration
#   fi
#     echo "Extracting $target for $split :"
#     python3 experiments/data/extract_targets.py \
#         --target $target \
#         --data_path /home/adamizdebski/files/data/datasets/guacamol/$split/smiles.txt \
#         --output /home/adamizdebski/files/data/datasets/guacamol/$split/$target.npy \
#         --n_workers 32
#   done
# done

# python3 experiments/data/extract_targets.py \
#         --target guacamol_mpo \
#         --data_path /home/adamizdebski/files/data/datasets/guacamol/train/10000/seed_0/smiles.txt \
#         --output /home/adamizdebski/files/data/datasets/guacamol/train/10000/seed_0/guacamol_mpo.npy \
#         --n_workers 32

python3 experiments/data/extract_targets.py \
        --target qed \
        --data_path /home/adamizdebski/files/data/datasets/guacamol/train/smiles.txt \
        --output /home/adamizdebski/files/data/datasets/guacamol/train/qed.npy \
        --n_workers 32
        
python3 experiments/data/extract_targets.py \
        --target qed \
        --data_path /home/adamizdebski/files/data/datasets/guacamol/val/smiles.txt \
        --output /home/adamizdebski/files/data/datasets/guacamol/val/qed.npy \
        --n_workers 32
        
python3 experiments/data/extract_targets.py \
        --target qed \
        --data_path /home/adamizdebski/files/data/datasets/guacamol/test/smiles.txt \
        --output /home/adamizdebski/files/data/datasets/guacamol/test/qed.npy \
        --n_workers 32
        