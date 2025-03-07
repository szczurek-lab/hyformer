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

from experiments.fibrosis_prediction.train import main as model_training_loop
from experiments.fibrosis_prediction.test import main as model_testing_loop


def main(args):
    _out_dir = args.out_dir

    val_metrics = {
        'loss': np.zeros(args.num_seeds),
        'auprc': np.zeros(args.num_seeds)
    }

    test_metrics = {
        'loss': np.zeros(args.num_seeds),
        'accuracy': np.zeros(args.num_seeds),
        'precision': np.zeros(args.num_seeds),
        'precision_100': np.zeros(args.num_seeds),
        'recall': np.zeros(args.num_seeds),
        'recall_100': np.zeros(args.num_seeds),
        'auroc': np.zeros(args.num_seeds),
        'auprc': np.zeros(args.num_seeds),
        'bedroc_20': np.zeros(args.num_seeds)
        }


    for seed in range(args.num_seeds):
        print(f"Running seed: {seed}")
        args.seed = seed
        args.out_dir = os.path.join(_out_dir, f"seed_{seed}")

        if os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):    
            print(f"Model already trained. Loading checkpoint...")
        
        else:
            # Create root directory
            os.makedirs(args.out_dir, exist_ok=True)
            
            # Infer number of epochs
            _predictive_max_iters = model_training_loop(args, max_iters='infer')
            print(f"Predictive iters set to: {_predictive_max_iters}")
        
            # Test intermediate model
            test_loss = model_testing_loop(args)
            print(f"Test loss (uncorrected): {test_loss}")
    
            val_loss = model_training_loop(args, max_iters=_predictive_max_iters)
            print(f"Best predictive validation loss with hparams: {val_loss}")


        # Test final model on validation set
        _test_metrics = model_testing_loop(args, split='val')
        for key, _ in test_metrics.items():
            test_metrics[key][seed] = _test_metrics[key]

        print(f"Test loss with hparams: {test_loss}")
        save_json(os.path.join(args.out_dir, "test_loss_aggregated.json"), {"mean": test_loss})

        # I run this on seed 0 to get the best hyperparameters
        # Then I run this on all seeds to get the test metrics


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
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of seeds to run the experiment with")
    args = parser.parse_args()
    main(args)
