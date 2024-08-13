#!/bin/bash

splits=("train" "val" "test")
targets=("plogp" "qed" "guacamol_mpo" "physchem")

for split in "${splits[@]}"
do
  for target in "${targets[@]}"
  do
     if [ "$split" == "train" ] && [ "$target" == "guacamol_mpo" ]; then
    continue # Skip the current iteration
  fi
    echo "Extracting $target for $split :"
    python3 experiments/data/extract_targets.py \
        --target $target \
        --data_path /home/adamizdebski/files/data/datasets/guacamol/$split/smiles.txt \
        --output /home/adamizdebski/files/data/datasets/guacamol/$split/$target.npy \
        --n_workers 32
  done
done
