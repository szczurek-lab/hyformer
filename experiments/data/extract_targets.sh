splits=(train test val)

target=$1

for split in ${splits[@]}; do

python3 experiments/data/extract_targets.py \
        --target $target \
        --output outputs/common/${split}/${target}.npy \
        --data_path data/guacamol/${split}/smiles.txt \
        --n_workers 32
done