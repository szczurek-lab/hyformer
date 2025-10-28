#!/bin/bash
# HyFormer Incremental SBS (Gumbeldore) sampling with SA oracle
#SBATCH --job-name=hyf-gumb-incsbs-sa
#SBATCH --account=hai_1077
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=/p/home/jusers/korkmaz1/juwels/hyformer/hyformer/output/%x.%j.out
#SBATCH --error=/p/home/jusers/korkmaz1/juwels/hyformer/hyformer/output/%x.%j.err
set -euo pipefail
# === micromamba ===
export PROJECT_ID=hai_1077
export BIN_DIR=/p/project1/$PROJECT_ID/$USER/bin
export MICROMAMBA_ROOT=/p/project1/$PROJECT_ID/$USER/micromamba_root

if [ ! -f "$BIN_DIR/micromamba" ]; then
    echo "micromamba not found at $BIN_DIR/micromamba" >&2
    exit 1
fi

echo "=== Hyformer Gumbeldore Incremental SBS (SA) ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Time: $(date)"
echo

# CUDA check
"$BIN_DIR/micromamba" run -n hyformer-neurips25 \
  python -c "import torch;print('CUDA:',torch.cuda.is_available(),'nGPU:',torch.cuda.device_count())" || {
  echo "CUDA check failed" >&2; exit 1; }

# Project source (this contains hyformer/conditional_sampling_gumbeldore/gumbeldore.py)
cd /p/home/jusers/korkmaz1/juwels/hyformer/hyformer

# Paths (from your self-improvement script)
MODEL_CONFIG=/p/project1/hai_1077/hyformer/korkmaz1/hyformer/configs/models/hyformer_downstream/gumbeldore/config.json
MODEL_CKPT=//p/home/jusers/korkmaz1/juwels/hyformer/hyformer/conditional_sampling_corrected_joint/sa/llama_backbone/max_epochs_10/decay_lr_true/batch_size_256/learning_rate_5e-4/weight_decay_1e-1/pooler_dropout_0.0/seed_0/ckpt.pt
TOKENIZER_CONFIG=/p/home/jusers/korkmaz1/juwels/hyformer/hyformer/configs/tokenizers/smiles/deepchem/config.json

# Sampling knobs
METHOD=inc_sbs
MAX_NEW_TOKENS=128
TEMPERATURE=1.0
TOP_K=0
BEAM_WIDTH=32
NUM_ROUNDS=10
ADV_CONSTS=(0.6)
MIN_TOP_P=0.9
target_values=(0.7)
LOSS=1minusAbsoluteError
invalid_penalty=-1.5
oracle_source=prediction
# Property-specific normalization parameters
QED_pred_mean=0.5538575
QED_pred_std=0.2140205
SA_pred_mean=0.7896584646658871
SA_pred_std=0.08999102410090047
LOGP_pred_mean=2.995084721452451
LOGP_pred_std=1.7325130267016382

# Define properties to test
properties=("sa")

for property_name in "${properties[@]}"; do
  for target_value in "${target_values[@]}"; do
    for ADV_CONST in "${ADV_CONSTS[@]}"; do
      echo "--- Running ${property_name} with target_value=${target_value}, advantage_constant=${ADV_CONST} ---"
      
      # Set property-specific normalization parameters
      if [ "$property_name" = "qed" ]; then
        pred_mean=$QED_pred_mean
        pred_std=$QED_pred_std
      elif [ "$property_name" = "sa" ]; then
        pred_mean=$SA_pred_mean
        pred_std=$SA_pred_std
      elif [ "$property_name" = "logp" ]; then
        pred_mean=$LOGP_pred_mean
        pred_std=$LOGP_pred_std
      fi
      
      "$BIN_DIR/micromamba" run -n hyformer-neurips25 \
        python -m scripts.gumbeldore_sampling.gumbeldore \
          --model_config "$MODEL_CONFIG" \
          --model_ckpt   "$MODEL_CKPT" \
          --tokenizer_config "$TOKENIZER_CONFIG" \
          --method "$METHOD" \
          --target_name "$property_name" \
          --target_value "$target_value" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          --temperature "$TEMPERATURE" \
          --top_k "$TOP_K" \
          --beam_width "$BEAM_WIDTH" \
          --num_rounds "$NUM_ROUNDS" \
          --advantage_constant "$ADV_CONST" \
          --min_top_p "$MIN_TOP_P" \
          --invalid_penalty "$invalid_penalty" \
          --oracle_source "$oracle_source" \
          --pred_mean "$pred_mean" \
          --pred_std "$pred_std" \
          --loss_function "$LOSS" \
            --out_csv /p/home/jusers/korkmaz1/juwels/hyformer/hyformer/output/samples_${property_name}_NumRounds"$NUM_ROUNDS"_BeamWidth"$BEAM_WIDTH"_Temp"$TEMPERATURE"_TopK"$TOP_K"_AdvConst"$ADV_CONST"_MinTopP"$MIN_TOP_P"_TargetValue"$target_value"_"$LOSS"_invalid_penalty"$invalid_penalty"_oracle_source"$oracle_source".csv
      done
  done
done

echo "Done."


