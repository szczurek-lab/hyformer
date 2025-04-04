#!/bin/bash

splits=("train" "val" "test")
targets=("plogp" "qed" "guacamol_mpo" "physchem")

for split in "${splits[@]}"
do
  for target in "${targets[@]}"
  do
    echo "Extracting $target for $split :"
    python3 experiments/data/extract_targets.py \
        --target $target \
        --data_path data/guacamol/$split/smiles.txt \
        --output data/guacamol/$split/$target.npy \
        --n_workers 32
  done
done
