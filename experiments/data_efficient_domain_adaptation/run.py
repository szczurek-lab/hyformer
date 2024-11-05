import os
import sys
import argparse
import optuna
import logging

from socket import gethostname
from functools import partial

from jointformer.utils.optuna import get_hparam_search_space, load_json, save_json

from experiments.data_efficient_domain_adaptation.train import main as model_training_loop
from experiments.data_efficient_domain_adaptation.test import main as model_validation_loop

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", "0"))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--fraction_train_dataset", type=float, default=1.)
    parser.add_argument("--model_seed", type=int, required=True)
    parser.add_argument("--hyperparameters_grid_filepath", type=str, default='experiments/data_efficient_domain_adaptation/hyperparameters_grid.json')
    parser.add_argument("--optuna_metric_direction", type=str, default='minimize')
    parser.add_argument("--optuna_n_trials", type=int, default=2)
    parser.add_argument("--optuna_n_jobs", type=int, default=1)
    parser.add_argument("--optuna_seed", type=int, default=42)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_seeds", type=int, default=4)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--destroy_ckpt", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--best_hparams_filename", type=str, default="best_hparams.json")
    parser.add_argument("--study_filename", type=str, default="study_results.csv")
    args = parser.parse_args()
    return args


def model_objective(trial, hyperparameters_grid, args):
    """
    Objective function for Optuna to optimize.

    Args:
        trial (optuna.trial.Trial): The trial for which to generate the search space.

    Returns:
        float: The value of the objective function.
    """
    # Generate hyperparameters for the trial
    hyperparams = get_hparam_search_space(trial, hyperparameters_grid)
    objective_value = model_training_loop(args, hyperparams)
    return objective_value


def find_best_params(args):
    # Load hyperparameters grid
    hyperparameters_grid = load_json(args.hyperparameters_grid_filepath)

    # Disable logging
    output_dir = args.out_dir
    args.out_dir = None 

    # Create actual objective function using partial - pass in the hyperparameters grid
    objective_func = partial(model_objective, hyperparameters_grid=hyperparameters_grid, args=args)

    # Create a study object
    study = optuna.create_study(direction=args.optuna_metric_direction, sampler=optuna.samplers.TPESampler(seed=args.seed))

    # Start the hyperparameter tuning
    study.optimize(objective_func, n_trials=args.optuna_n_trials, n_jobs=args.optuna_n_jobs)
    study_df = study.trials_dataframe()
    args.out_dir = output_dir

    # Save study dataframe
    study_df.to_csv(os.path.join(args.out_dir, args.study_filename), index=False)
    
    # Save best params
    console.info(f"Saving best hparams to {args.best_hparams_filename}.")
    save_json(os.path.join(args.out_dir, args.best_hparams_filename), study.best_params)

    return None


def load_best_params(args):
    console.info(f"Hparams loaded from {args.best_hparams_filename}.")
    return load_json(os.path.join(args.out_dir, args.best_hparams_filename))


def main(args):

    # set root directory
    root_dir = args.out_dir

    # Iterate over different trials
    for seed in range(args.num_seeds):
        
        # Create output directory for model training
        output_dir = os.path.join(root_dir, f"seed_{seed}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set seed and output directory
        args.seed = seed
        args.out_dir = output_dir
        
        # Find best hyperparameters
        if not os.path.exists(os.path.join(args.out_dir, args.best_hparams_filename)):
            find_best_params(args)

        # Load best hyperparameters
        hparams = load_best_params(args)

        # Train and Test model
        args.out_dir = output_dir
        model_training_loop(args, hparams)
        model_validation_loop(args, hparams)

if __name__ == "__main__":
    args = parse_args()
    main(args)
