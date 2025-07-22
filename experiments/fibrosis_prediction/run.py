import os
import argparse

import numpy as np

from hyformer.utils.optuna import save_json

from experiments.fibrosis_prediction.train import main as model_training_loop
from experiments.fibrosis_prediction.test import main as model_testing_loop


def main(args):
    
    # Set root directory
    args.seed = args.dataset_seed    
    args.out_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)
    
    # Terminate, if model already trained
    if os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):    
        print(f"Model already trained for seed {args.dataset_seed}.")
        return None
    
    # Infer number of epochs
    _out_dir = args.out_dir
    _path_to_pretrained = args.path_to_model_ckpt

    args.out_dir = os.path.join(args.out_dir, f"infer_training_epochs")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)
    
    if args.path_to_generative_trainer_config is not None:
        args.path_to_trainer_config = args.path_to_generative_trainer_config
        _generative_max_iters = model_training_loop(args, max_iters='infer_generation')
        print(f"Generative max iters: {_generative_max_iters}")
        args.path_to_model_ckpt = os.path.join(args.out_dir, 'ckpt.pt')
    args.path_to_trainer_config = args.path_to_predictive_trainer_config
    _predictive_max_iters = model_training_loop(args, max_iters='infer')
    print(f"Predictive max iters: {_predictive_max_iters}")
    
    # Test intermediate model
    test_loss = model_testing_loop(args)
    print(f"Test loss (uncorrected): {test_loss}")
    
    args.out_dir = _out_dir
    args.path_to_model_ckpt = _path_to_pretrained

    # Train
    if args.path_to_generative_trainer_config is not None:
        args.path_to_trainer_config = args.path_to_generative_trainer_config
        val_loss = model_training_loop(args, max_iters=_generative_max_iters)
        print(f"Best generative validation loss with hparams: {val_loss}")
        args.path_to_model_ckpt = os.path.join(args.out_dir, 'ckpt.pt')
    args.path_to_trainer_config = args.path_to_predictive_trainer_config
    val_loss = model_training_loop(args, max_iters=_predictive_max_iters)
    print(f"Best predictive validation loss with hparams: {val_loss}")

    # Test
    if not hasattr(args, 'path_to_trainer_config'):
        args.path_to_trainer_config = args.path_to_predictive_trainer_config
    test_loss = model_testing_loop(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hi experiment.")
    parser.add_argument("--out-dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--data_dir", type=str, nargs='?', help="Path to the data directory")
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_generative_trainer_config", type=str, default=None)
    parser.add_argument("--path_to_predictive_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--decay_lr", type=lambda x: (str(x).lower() == 'true'), default=None, help="Dropout for the model")
    parser.add_argument("--batch_size", type=int, default=None, help="Dropout for the model")
    parser.add_argument("--learning_rate", type=float, default=None, help="Dropout for the model")
    parser.add_argument("--weight_decay", type=float, default=None, help="Dropout for the model")
    parser.add_argument("--max_epochs", type=int, default=None, help="Dropout for the model")
    parser.add_argument("--pooler_dropout", type=float, default=None, help="Dropout for the model")
    parser.add_argument("--patience", type=int, default=None, help="Patience for the model")
    parser.add_argument("--dataset_seed", type=int, default=1, help="Number of seeds to run the experiment with")
    parser.add_argument("--model_seed", type=int, default=1, help="Number of seeds to run the experiment with")
    args = parser.parse_args()
    main(args)
