""" Predict target properties.

Usage:

python scripts/predict.py \
    --path_to_sequence_file data/raw/sequences.csv \
    --path_to_output_file results/processed/predictions.csv \
    --path_to_tokenizer_config configs/tokenizers/smiles/deepchem/config.json \
    --path_to_model_config configs/models/hyformer_downstream/50M/config.json \
    --path_to_model_ckpt <PATH_TO_MODEL_CKPT> \
    --path_to_sequence_column smiles \
    --device cuda:0 \
    --batch_size 128 \
    --seed 1337 \
    --task_type classification \
    --num_tasks 1

"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd

from typing import Optional, List

from hyformer.configs import TokenizerConfig, ModelConfig
from hyformer.utils import AutoTokenizer
from hyformer import AutoModel
from hyformer.utils.runtime import set_seed, get_device

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--path_to_sequence_file", type=str, required=True)
    parser.add_argument("--path_to_output_file", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--path_to_sequence_column", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def _load_sequences(path_to_sequences_file: str, path_to_sequences_column: Optional[str] = None) -> List[str]:
    if path_to_sequences_file.endswith('.smiles') or path_to_sequences_file.endswith('.txt'):
        with open(path_to_sequences_file, 'r') as f:
            sequences = f.readlines()
        return sequences
    elif path_to_sequences_file.endswith('.csv'):
        assert path_to_sequences_column is not None, "Path to sequences column is required for CSV files."
        df = pd.read_csv(path_to_sequences_file)
        assert path_to_sequences_column in pd.read_csv(path_to_sequences_file).columns, \
            f"Column {path_to_sequences_column} not found in {path_to_sequences_file}. Available columns: {pd.read_csv(path_to_sequences_file).columns.tolist()}"
        return df[path_to_sequences_column].tolist()
    elif path_to_sequences_file.endswith('.npz'):
        return np.load(path_to_sequences_file)[path_to_sequences_column].tolist()
    else:
        raise ValueError(f"Unsupported file extension: {path_to_sequences_file}")


def _save(sequences: list[str], predictions: np.ndarray, path_to_output_file: str) -> None:
    if path_to_output_file.endswith('.csv'):
        columns = ['prediction'] if predictions.shape[1] == 1 else [f'prediction_{i}' for i in range(predictions.shape[1])]
        df = pd.DataFrame(predictions, columns=columns)
        df['sequence'] = sequences
        df.to_csv(path_to_output_file, index=False)
        logger.info(f"Saved predictions of shape {predictions.shape} to {path_to_output_file}")
        return None
    elif path_to_output_file.endswith('.npz'):
        _sequence_dtype = f"<U{max(len(sequence) for sequence in sequences)}"
        np.savez(path_to_output_file, prediction=predictions, sequence=np.array(sequences, dtype=_sequence_dtype))
        logger.info(f"Saved predictions of shape {predictions.shape} to {path_to_output_file}")
        return None
    else:
        raise ValueError(f"Unsupported file extension: {path_to_output_file}")
    return None


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = get_device()

    # Load dataset
    sequences = _load_sequences(args.path_to_sequence_file, args.path_to_sequence_column)

    # Load tokenizer
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    # Load model
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    model = AutoModel.from_config(model_config, downstream_task=args.task_type, num_tasks=args.num_tasks)
    model.load_pretrained(args.path_to_model_ckpt)
    model = model.to_predictor(tokenizer, args.batch_size, device)

    # Predict
    predictions = model.predict(sequences)

    # Save predictions
    _save(sequences, predictions, args.path_to_output_file)
    return None


if __name__ == "__main__":
    main()
    