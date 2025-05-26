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


NUM_SAMPLES = 30000


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

    # Generate samples
    generated_samples = []
    num_iterations = NUM_SAMPLES // args.batch_size if NUM_SAMPLES % args.batch_size == 0 else NUM_SAMPLES // args.batch_size + 1

    for _ in tqdm(range(num_iterations)):
 
        _samples_idx = model.generate(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
            ).detach().cpu()
        _samples = tokenizer.decode(_samples_idx)
        generated_samples.extend(_samples)
        
    if len(generated_samples) > NUM_SAMPLES:
        generated_samples = generated_samples[:NUM_SAMPLES]

    # 4. Save generated samples and properties
    pd.DataFrame(
        {
            "smiles": generated_samples,
        }).to_csv(
            os.path.join(args.out_dir, f"unconditionally_generated_samples_temperature={args.temperature}_topk={args.top_k}.csv"),
            index=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Optuna Worker")
    parser.add_argument("--out_dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--seed", type=int, default=1337, help="Seed for Optuna study")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for sampling")
    parser.add_argument("--top_k", type=int, default=25, help="Number of generations")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
