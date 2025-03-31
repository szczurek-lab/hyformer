# Imports

import os
import torch
import argparse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig

from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel

from jointformer.utils.runtime import set_seed
from jointformer.utils.chemistry import is_valid

from experiments.conditional_sampling.utils import get_logp, get_qed, get_sa


def main(args):
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    # Load model
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    model = AutoModel.from_config(model_config, downstream_task="regression", num_tasks=1)
    model.load_pretrained(args.path_to_model_ckpt)
    model.to(device)
    model.eval()

    # Load oracle
    if args.target_name == "qed":
        oracle = get_qed
    elif args.target_name == "logp":
        oracle = get_logp
    elif args.target_name == "sa":
        oracle = get_sa
    else:
        raise ValueError(f"Unknown target name: {args.target_name}")

    # Generate samples
    generated_samples = []
    generated_properties = []
    oracle_values = []
    is_valid_molecule = []
    generated_idx = []

    for _idx in tqdm(range(args.num_generation_iters)):

        # 1. Generate K-many unique samples 
        samples = []
        while len(samples) < args.k:
            _samples_idx = model.generate(
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                temperature=args.temperature,
                top_k=args.generation_k,
                device=device
                ).detach().cpu()
            _samples = tokenizer.decode(_samples_idx)
            samples.extend(_samples)
            samples = list(set(samples) - set(generated_samples))
        samples = samples[:args.k]

        # 2. Compute properties with a surrogate model
        properties = []
        for i in range(0, len(samples), args.batch_size):
            _batch = samples[i:i+args.batch_size]
            _inputs = tokenizer(_batch, task='prediction')
            _inputs.to(device)
            _properties = model.predict(**_inputs).detach().cpu()
            properties.extend(_properties.numpy().flatten().tolist()) 
        
        # 3. Compute oracle values
        _oracle_values = [oracle(s) for s in samples]

        # 4. Filter out invalid samples
        _valid_samples = [is_valid(s) for s in samples]

        # 5. Generate sample idx
        idx = [_idx for _ in range(len(samples))]

        generated_samples.extend(samples)
        generated_properties.extend(properties)
        oracle_values.extend(_oracle_values)
        is_valid_molecule.extend(_valid_samples)
        generated_idx.extend(idx)  
        
        # 4. Save generated samples and properties
        pd.DataFrame(
            {
                "idx": generated_idx,
                "smiles": generated_samples,
                "properties": generated_properties,
                "oracle_values": oracle_values,
                "is_valid_smiles": is_valid_molecule
            }).to_csv(
                os.path.join(args.out_dir, f"generated_samples_{args.target_name}.csv"),
                index=False
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Optuna Worker")
    parser.add_argument("--target_name", type=str, default=None, help="Target name for the model", choices=["qed", "logp", "sa"])
    parser.add_argument("--out-dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--data_dir", type=str, nargs='?', help="Path to the data directory")
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--debug_only", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=1337, help="Seed for Optuna study")
    parser.add_argument("--num_generation_iters", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for sampling")
    parser.add_argument("--generation_k", type=int, default=25, help="Number of generations")
    parser.add_argument("--k", type=int, default=1024, help="Number of samples to generate")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
