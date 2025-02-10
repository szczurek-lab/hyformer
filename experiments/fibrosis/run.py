import os
import optuna
import argparse
import torch 

import numpy as np

from functools import partial

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig

from jointformer.models.auto import AutoModel
from jointformer.trainers.trainer_fixed import Trainer

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.utils.optuna import load_json, get_hparam_search_space, save_json

from experiments.hi.train import main as model_training_loop
from experiments.hi.test import main as model_testing_loop


def main(args):
        
    if not os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):    
        
        # Create root directory
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=False)
        
        # Infer number of epochs
        _predictive_max_iters = model_training_loop(args, max_iters='infer')
        print(f"Predictive max iters: {_predictive_max_iters}")
        
        # Test intermediate model
        test_loss = model_testing_loop(args)
        print(f"Test loss (uncorrected): {test_loss}")
    
        val_loss = model_training_loop(args, max_iters=_predictive_max_iters)
        print(f"Best predictive validation loss with hparams: {val_loss}")

    else:
        print(f"Model already trained. Loading checkpoint...")
    
    test_loss = model_testing_loop(args)
    print(f"Test loss with hparams: {test_loss}")
    save_json(os.path.join(args.out_dir, "test_loss_aggregated.json"), {"mean": test_loss})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hi experiment.")
    parser.add_argument("--out-dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--data_dir", type=str, nargs='?', help="Path to the data directory")
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, default=None)
    parser.add_argument("--path_to_generative_trainer_config", type=str, default=None)
    parser.add_argument("--path_to_predictive_trainer_config", type=str, default=None)
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for the model")
    parser.add_argument("--batch_size", type=int, default=None, help="Learning rate for the model")
    parser.add_argument("--patience", type=int, default=None, help="Patience for the model")
    args = parser.parse_args()
    main(args)
