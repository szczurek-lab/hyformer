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

from experiments.data_efficient_domain_adaptation.train import main as model_training_loop
from experiments.data_efficient_domain_adaptation.test import main as model_testing_loop


def objective(trial, hparams_grid, train_dataset, val_dataset, test_dataset, tokenizer, model_config, trainer_config,
              debug_only, downstream_task_type, num_downstream_tasks, path_to_model_ckpt, metric, direction):

    try:
        hparams = get_hparam_search_space(trial, hparams_grid)

        # Update configs // TODO: Refactor this to optuna_utils 
        for key, value in hparams.items():
            if key == 'scale_beta' and value:
                trainer_config.beta1 = 0.9
                trainer_config.beta1 = 0.999
            if key in model_config.__dict__.keys():
                model_config[key] = value
                print(f"Setting {key} to {value}")
            if key in trainer_config.__dict__.keys():
                trainer_config[key] = value
                print(f"Setting {key} to {value}")
                if key == 'learning_rate':
                    trainer_config['min_lr'] = 0.1 * value
            if key == 'generation_task':
                trainer_config['tasks'] = {"prediction": 100 - value, "generation": value}
                trainer_config._normalize_task_probabilities()
            
        # and adjust trainer config to dataset size
        trainer_config.correct_for_num_train_examples(num_train_examples=len(train_dataset)) 

        # Debug
        if debug_only:
            trainer_config.max_iters = 2
            trainer_config.batch_size = 2
            trainer_config.eval_iters = 1
            trainer_config.eval_interval = 1
            trainer_config.log_interval = 1

        # Init
        model = AutoModel.from_config(model_config, downstream_task=downstream_task_type, num_tasks=num_downstream_tasks)
        device = torch.device('cuda:0')
        trainer = Trainer(out_dir=None, seed=1337, config=trainer_config, model=model, train_dataset=train_dataset, eval_metric='prediction',
                          val_dataset=val_dataset, test_dataset=test_dataset, tokenizer=tokenizer, test_metric=metric, device=device, patience=args.patience)

        # Load
        if path_to_model_ckpt is not None:
            if not os.path.exists(path_to_model_ckpt):
                raise ValueError(f"Model checkpoint {path_to_model_ckpt} does not exist.")
            trainer.resume_from_file(path_to_model_ckpt)

        trainer.train()
        return trainer._optuna_loss
    except:
        print(f"Trial: {hparams} diverged...")
        if direction == 'minimize':
            return 1e-9
        elif direction == 'maximize':
            return 0.0
        else:
            return ValueError


def find_best_hparams(args):

    # Load data
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    test_dataset = AutoDataset.from_config(dataset_config, split='test', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    
    trainer_config.max_epochs = min(trainer_config.max_epochs, args.optuna_max_epochs)  # set max_epochs
    print(f"Max epochs set to {trainer_config.max_epochs}")

    # Attempt to load the study; if it doesn't exist, create it
        
    # set direction
    if dataset_config.task_metric in ['rmse']:
        direction = 'minimize'
    elif dataset_config.task_metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        direction = 'maximize'
    else:
        raise ValueError(f"Invalid metric: {dataset_config.task_metric}")

    # set sampler
    if args.sampler == 'grid':
        search_space = load_json(args.search_space_filepath)
        print("Search space:", search_space)
        assert search_space is not None
        sampler = optuna.samplers.GridSampler(search_space=search_space)
    elif args.sampler == 'TPE':
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        raise ValueError(f"Invalid sampler: {args.sampler}")
    
    # create study
    study = optuna.create_study(study_name=args.study_name, storage=args.storage, direction=direction, sampler=sampler, load_if_exists=args.load_if_exists)

    # Optimize
    hparams_grid = load_json(args.hparams_grid_filepath)
    print("Hyperparameters grid:", hparams_grid)
    objective_function = partial(objective, hparams_grid=hparams_grid, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                                tokenizer=tokenizer, direction=direction, model_config=model_config, trainer_config=trainer_config, debug_only=args.debug_only,
                                metric=dataset_config.task_metric, downstream_task_type=dataset_config.task_type, num_downstream_tasks=dataset_config.num_tasks,
                                path_to_model_ckpt=args.path_to_model_ckpt)
    study.optimize(objective_function, n_trials=args.optuna_n_trials, n_jobs=args.optuna_n_jobs) # works for hasattr(self.model, 'predict')
    
    # Save the study
    study_results_filepath = os.path.join(args.out_dir, "study_results.csv")
    study.trials_dataframe().to_csv(study_results_filepath, index=False)
    print(f"Study results saved to {study_results_filepath}.")

    # Save the best hyperparameters
    best_hparams_filepath = os.path.join(args.out_dir, args.best_hparams_filename)
    save_json(best_hparams_filepath, study.best_params)
    print(f"Best hyperparameters saved to {best_hparams_filepath}.")


def load_best_hparams(args):
    best_hparams_filepath = os.path.join(args.out_dir, args.best_hparams_filename)
    return load_json(best_hparams_filepath)


def main(args):
    
    # Create root directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)

    # Find best hyperparameters
    hparams = None
    if args.search_space_filepath is not None:
        if not os.path.exists(os.path.join(args.out_dir, args.best_hparams_filename)):
            print("Finding best hyperparameters...")
            find_best_hparams(args)
    
    # Load best hyperparameters
    if os.path.exists(os.path.join(args.out_dir, args.best_hparams_filename)):
        print("Loading best hyperparameters...")
        hparams = load_best_hparams(args)
    else:
        print("Defaulting to hyperparameters from the config file.")

    # Train and Test model
    root_dir = args.out_dir
    test_loss_arrary = np.zeros(shape=(3,))
    for seed in [0, 1, 2]:
        print(f"Training and testing model with best hyperparameters for seed {seed}...")
        args.seed = seed
        args.out_dir = os.path.join(root_dir, f"seed_{seed}")
        if not os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):    
            if args.path_to_generative_trainer_config is not None:
                _predictive_trainer = args.path_to_trainer_config
                _out_dir = args.out_dir
                args.path_to_trainer_config = args.path_to_generative_trainer_config
                args.out_dir = os.path.join(_out_dir, "generative")
                val_loss = model_training_loop(args, hparams)
                args.path_to_model_ckpt = os.path.join(args.out_dir, "ckpt.pt")
                args.path_to_trainer_config = _predictive_trainer
                args.out_dir = _out_dir

            val_loss = model_training_loop(args, hparams)
            print(f"Best validation loss with hparams: {val_loss}")
        else:
            print(f"Model already trained for seed {seed}. Loading checkpoint...")
        test_loss = model_testing_loop(args, hparams)
        print(f"Test loss with hparams: {test_loss}")
        test_loss_arrary[seed] = test_loss
        args.out_dir = root_dir

    print(f"Test loss array: {test_loss_arrary}")
    print(f"Mean test loss: {np.mean(test_loss_arrary)}")
    print(f"Std test loss: {np.std(test_loss_arrary)}")
    print(f"Latex entry regression: {round(np.mean(test_loss_arrary), 3)}$\pm${round(np.std(test_loss_arrary), 3)}")
    print(f"Latex entry classification: {round(np.mean(test_loss_arrary) * 100, 1)}({round(np.std(test_loss_arrary) * 100, 1)})")
    
    save_json(os.path.join(args.out_dir, "test_loss_aggregated.json"), {"mean": np.mean(test_loss_arrary), "std": np.std(test_loss_arrary), "se": np.std(test_loss_arrary) / np.sqrt(3)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Optuna Worker")
    parser.add_argument("--out-dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--data_dir", type=str, nargs='?', help="Path to the data directory")
    parser.add_argument("--study_name", type=str, default=None, help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None, help="Database URL for Optuna study")
    parser.add_argument("--optuna-n-trials", type=int, default=None, help="Number of trials to run in this instance")
    parser.add_argument("--optuna-n-jobs", type=int, default=1, help="Number of parallel jobs to run")
    parser.add_argument("--seed", type=int, default=0, help="Seed for Optuna study")
    parser.add_argument("--sampler", type=str, default='grid', choices=['grid', 'TPE'], help="Sampler to use for Optuna study")
    parser.add_argument("--search_space_filepath", type=str, default=None, help="Path to the JSON file containing the search space")
    parser.add_argument("--hparams_grid_filepath", type=str, default=None, help="Path to the JSON file containing the hyperparameters grid")
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_generative_trainer_config", type=str, default=None)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--debug_only", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--optuna_max_epochs", type=int, default=20, help="Maximum number of epochs to train models for hparams search.")
    parser.add_argument("--best_hparams_filename", type=str, default="best_hparams.json", help="Filename to save best hyperparameters to.")
    parser.add_argument("--load_if_exists", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--destroy_ckpt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--adjust_dataset_seed", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_metric", type=str, default=None, help="Evaluation metric for the model")
    parser.add_argument("--decay_lr", type=lambda x: (str(x).lower() == 'true'), default=None, help="Dropout for the model")
    parser.add_argument("--batch_size", type=int, default=None, help="Dropout for the model")
    parser.add_argument("--learning_rate", type=float, default=None, help="Dropout for the model")
    parser.add_argument("--weight_decay", type=float, default=None, help="Dropout for the model")
    parser.add_argument("--max_epochs", type=int, default=None, help="Dropout for the model")
    parser.add_argument("--pooler_dropout", type=float, default=None, help="Dropout for the model")
    parser.add_argument("--patience", type=int, default=None, help="Patience for the model")
    args = parser.parse_args()
    main(args)
