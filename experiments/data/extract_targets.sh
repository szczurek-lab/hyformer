target=$1
split=$2


python3 experiments/data/extract_targets.py \
        --target $target \
        --output outputs/jointformer/${split}/${target}.npy \
        --data_path /home/jano1906/git/jointformer/data/guacamol/${split}/smiles.txt \
        --n_workers 32 