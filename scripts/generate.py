""" Featurize a list of sequences.

Usage:

python scripts/generate.py \
    --path_to_output_file data/processed/generated_samples.csv \
    --path_to_tokenizer_config configs/tokenizers/smiles/deepchem/config.json \
    --path_to_model_config configs/models/hyformer/50M/config.json \
    --path_to_model_ckpt results/smiles/50M/joint/ckpt.pt \
    --device cuda:0 \
    --batch_size 256 \
    --seed 1337 \
    --temperature 0.9 \
    --top_k 25

"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd

from typing import Optional, List

from hyformer import AutoModel
from hyformer.models.base import Generator
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


def _save(sequences: list[str], path_to_output_file: str) -> None:
    os.makedirs(os.path.dirname(path_to_output_file), exist_ok=True)
    if path_to_output_file.endswith('.csv'):
        df = pd.DataFrame(sequences, columns=['sequence'])
        df.to_csv(path_to_output_file, index=False)
        logger.info(f"Saved sequences of shape {len(sequences)} to {path_to_output_file}")
        return None
    elif path_to_output_file.endswith('.npz'):
        _sequence_dtype = f"<U{max(len(sequence) for sequence in sequences)}"
        np.savez(path_to_output_file, sequence=np.array(sequences, dtype=_sequence_dtype))
        logger.info(f"Saved sequences of shape {len(sequences)} to {path_to_output_file}")
        return None
    elif path_to_output_file.endswith('.smiles') or path_to_output_file.endswith('.txt'):
        with open(path_to_output_file, 'w') as f:
            for i, sequence in enumerate(sequences):
                if i < len(sequences) - 1:
                    f.write(sequence + '\n')
                else:
                    f.write(sequence)
        logger.info(f"Saved sequences of shape {len(sequences)} to {path_to_output_file}")
        return None
    else:
        raise ValueError(f"Unsupported file extension: {path_to_output_file}")
    return None


def _load_tokenizer(path_to_tokenizer_config: str) -> BaseTokenizer:
    tokenizer_config = TokenizerConfig.from_config_file(path_to_tokenizer_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    return tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_output_file", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=10000)
    return parser.parse_args()


def main():
    args = _parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    tokenizer = _load_tokenizer(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    model = AutoModel.from_config(model_config)
    model.load_pretrained(args.path_to_model_ckpt)
    generator = model.to_generator(tokenizer, args.batch_size, args.temperature, args.top_k, device)
    sequences = generator.generate(args.num_samples)
    _save(sequences, args.path_to_output_file)
    return None
    

if __name__ == "__main__":
    main()
