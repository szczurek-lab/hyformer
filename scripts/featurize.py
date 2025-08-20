""" Featurize a list of sequences.

Usage:

python scripts/featurize.py \
    --path_to_sequence_file data/raw/sequences.csv \
    --path_to_output_file data/processed/embeddings.npz \
    --path_to_tokenizer_config configs/tokenizers/smiles/deepchem/config.json \
    --path_to_model_config configs/models/hyformer/50M/config.json \
    --path_to_model_ckpt <PATH_TO_MODEL_CKPT> \
    --path_to_sequence_column smiles \
    --device cuda:0 \
    --batch_size 128 \
    --seed 1337

"""

import sys
import logging
import argparse

import numpy as np
import pandas as pd

from typing import Optional, List

from hyformer import AutoModel
from hyformer.models.base import Encoder
from hyformer.utils import set_seed, get_device, AutoTokenizer
from hyformer.configs import TokenizerConfig, ModelConfig
from hyformer.utils.tokenizers.base import BaseTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)

    
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


def _save(sequences: list[str], embeddings: np.ndarray, path_to_output_file: str) -> None:
    _sequence_dtype = f"<U{max(len(sequence) for sequence in sequences)}"
    np.savez(path_to_output_file, embedding=embeddings, sequence=np.array(sequences, dtype=_sequence_dtype))
    logger.info(f"Saved embeddings of shape {embeddings.shape} to {path_to_output_file}")
    return None


def _load_tokenizer(path_to_tokenizer_config: str) -> BaseTokenizer:
    tokenizer_config = TokenizerConfig.from_config_file(path_to_tokenizer_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    return tokenizer


def _load_featurizer(tokenizer: BaseTokenizer, path_to_model_config: str, path_to_model_ckpt: str, device: str, batch_size: int) -> Encoder:
    model_config = ModelConfig.from_config_file(path_to_model_config)
    model = AutoModel.from_config(model_config)
    model.load_pretrained(path_to_model_ckpt)
    model.to(device)
    model.eval()
    return model.to_encoder(tokenizer, batch_size, device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_sequence_file", type=str, required=True)
    parser.add_argument("--path_to_output_file", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, required=True)
    parser.add_argument("--path_to_sequence_column", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    sequences = _load_sequences(args.path_to_sequence_file, args.path_to_sequence_column)
    tokenizer = _load_tokenizer(args.path_to_tokenizer_config)
    featurizer = _load_featurizer(tokenizer, args.path_to_model_config, args.path_to_model_ckpt, device, args.batch_size)
    embeddings = featurizer.encode(sequences)
    _save(sequences, embeddings, args.path_to_output_file)
    return None
    
if __name__ == "__main__":
    main()
