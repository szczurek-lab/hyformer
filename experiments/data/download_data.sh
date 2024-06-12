#!/bin/bash

python experiments/data/download_data.py \
    --data_dir /lustre/groups/aih/jointformer/data \
    --seed 0 \
    --path_to_task_config configs/tasks/guacamol/physchem \
    