import os
import argparse

import json
import numpy as np

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.trainer import TrainerConfig

from hyformer.models.auto import AutoModel
from hyformer.trainers.trainer import Trainer

from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.auto import AutoTokenizer

from scripts.finetune.train import main as model_training_loop
from scripts.finetune.test import main as model_testing_loop


def main(args):
    
    _master_ckpt = args.path_to_model_ckpt
    _root_dir = args.out_dir
    test_loss_arrary = np.zeros(shape=(3,))

    # Iterate over seeds
    for dataset_seed in [0, 1, 2]:
        
        args.out_dir = os.path.join(_root_dir, f"seed_{dataset_seed}")
        args.seed = dataset_seed
        _path_to_pretrained = _master_ckpt
        print(f"Training model with seed {dataset_seed}...")

        # Train model
        if not os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):    
            
            _out_dir = args.out_dir
            args.out_dir = os.path.join(args.out_dir, f"infer_training_epochs")
            args.path_to_model_ckpt = _path_to_pretrained

            # Create root directory
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir, exist_ok=False)
            
            # Infer number of epochs
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
            
            # if args.path_to_model_ckpt != _master_ckpt and args.path_to_generative_trainer_config is not None:
            #     os.remove(args.path_to_model_ckpt)

            if args.path_to_generative_trainer_config is not None:
                args.path_to_model_ckpt = _path_to_pretrained
                args.path_to_trainer_config = args.path_to_generative_trainer_config
                val_loss = model_training_loop(args, max_iters=_generative_max_iters)
                print(f"Best generative validation loss with hparams: {val_loss}")
                args.path_to_model_ckpt = os.path.join(args.out_dir, 'ckpt.pt')
            args.path_to_trainer_config = args.path_to_predictive_trainer_config
            val_loss = model_training_loop(args, max_iters=_predictive_max_iters)
            print(f"Best predictive validation loss with hparams: {val_loss}")
        else:
            print(f"Model already trained for seed {dataset_seed}. Loading checkpoint for ...")
        
        if not hasattr(args, 'path_to_trainer_config'):
            args.path_to_trainer_config = args.path_to_predictive_trainer_config
        test_loss = model_testing_loop(args)
        print(f"Test loss with hparams: {test_loss}")
        test_loss_arrary[dataset_seed] = test_loss

    print(f"Test loss array: {test_loss_arrary}")
    print(f"Mean test loss: {np.mean(test_loss_arrary)}")
    print(f"Std test loss: {np.std(test_loss_arrary)}")
    print(f"Latex entry: {round(np.mean(test_loss_arrary), 3)}$\pm${round(np.std(test_loss_arrary), 3)}")
    
    save_json(os.path.join(_root_dir, "test_loss_aggregated.json"), {"mean": np.mean(test_loss_arrary), "std": np.std(test_loss_arrary), "se": np.std(test_loss_arrary) / np.sqrt(3)})


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


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
    args = parser.parse_args()
    main(args)
