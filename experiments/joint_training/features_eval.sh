models=(moler jointformer)
targets=(plogp qed)
evals=(eval_linear)

for model in ${models[@]}; do
    train_data_path="outputs/${model}/train/guacamol_v1_features.npy"
    val_data_path="outputs/${model}/val/guacamol_v1_features.npy"
    test_data_path="outputs/${model}/test/guacamol_v1_features.npy"

    for target in ${targets[@]}; do
        train_target_path="outputs/common/train/${target}.npy"
        val_target_path="outputs/common/val/${target}.npy"
        test_target_path="outputs/common/test/${target}.npy"
        output_dir="outputs/${model}/runs"
        if [[ $target -eq "qed" ]]; then
            lr=0.001
        elif [[ $target -eq "plogp" ]]; then 
            lr=0.0001
            #lr=0.001
        else
            lr=""
        fi

        python3 experiments/joint_training/features_eval.py \
            --train_data_path=$train_data_path \
            --val_data_path=$val_data_path \
            --test_data_path=$test_data_path \
            --train_target_path=$train_target_path \
            --val_target_path=$val_target_path \
            --test_target_path=$test_target_path \
            --output_dir=$output_dir \
            --target=$target \
            --lr=$lr \
            --evals ${evals[@]}


    done
done