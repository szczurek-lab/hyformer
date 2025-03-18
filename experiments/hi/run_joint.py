import os
import optuna
import argparse
import torch 

import numpy as np

from functools import partial

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.trainer import TrainerConfig

from hyformer.models.auto import AutoModel
from hyformer.trainers.trainer import Trainer

from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.utils.optuna import load_json, get_hparam_search_space, save_json

from experiments.hi.train import main as model_training_loop
from experiments.hi.test import main as model_testing_loop


def objective(trial, hparams_grid, train_dataset, val_dataset, tokenizer, model_config, trainer_config,
              debug_only, downstream_task_type, num_downstream_tasks, path_to_model_ckpt, metric):
    hparams = get_hparam_search_space(trial, hparams_grid)

    # Update configs // TODO: Refactor this to optuna_utils 
    for key, value in hparams.items():
        if key in model_config.__dict__.keys():
            setattr(model_config, key, value)
        if key in trainer_config.__dict__.keys():
            setattr(trainer_config, key, value)
        if key == 'generation_task':
            trainer_config.tasks = {"prediction": 100 - value, "generation": value}
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
    model = AutoModel.from_config(model_config, downstream_task=downstream_task_type, num_tasks=num_downstream_tasks, hidden_dim=256)
    device = torch.device('cuda:0')
    trainer = Trainer(out_dir=None, seed=args.seed, config=trainer_config, model=model, train_dataset=train_dataset,
                      val_dataset=val_dataset, test_dataset=val_dataset, tokenizer=tokenizer, test_metric=metric, device=device)

    # Load
    if path_to_model_ckpt is not None:
        if not os.path.exists(path_to_model_ckpt):
            raise ValueError(f"Model checkpoint {path_to_model_ckpt} does not exist.")
        trainer.resume_from_file(path_to_model_ckpt)

    trainer.train()
    return trainer._optuna_loss


def find_best_hparams(args):

    # Load data
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    
    trainer_config.max_epochs = max(trainer_config.max_epochs, args.optuna_max_epochs)  # set max_epochs

    # Attempt to load the study; if it doesn't exist, create it
        
    # set direction
    if dataset_config.evaluation_metric in ['rmse']:
        direction = 'minimize'
    elif dataset_config.evaluation_metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        direction = 'maximize'
    else:
        raise ValueError(f"Invalid metric: {dataset_config.evaluation_metric}")

    # set sampler
    if args.sampler == 'grid':
        search_space = load_json(args.search_space_filepath)
        print("Search space:", search_space)
        assert search_space is not None
        sampler = optuna.samplers.GridSampler(search_space=search_space, seed=args.seed)
    elif args.sampler == 'TPE':
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        raise ValueError(f"Invalid sampler: {args.sampler}")
    
    # create study
    study = optuna.create_study(study_name=args.study_name, storage=args.storage, direction=direction, sampler=sampler, load_if_exists=args.load_if_exists)

    # Optimize
    hparams_grid = load_json(args.hparams_grid_filepath)
    objective_function = partial(objective, hparams_grid=hparams_grid, train_dataset=train_dataset, val_dataset=val_dataset, tokenizer=tokenizer,
                                 model_config=model_config, trainer_config=trainer_config, debug_only=args.debug_only, metric=dataset_config.evaluation_metric,
                                 downstream_task_type=dataset_config.task_type, num_downstream_tasks=dataset_config.num_tasks, path_to_model_ckpt=args.path_to_model_ckpt)
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
