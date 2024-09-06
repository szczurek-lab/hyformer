import os
import sys
import logging
import argparse

from socket import gethostname

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.auto import AutoLogger

from jointformer.trainers.trainer import Trainer

from jointformer.utils.runtime import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp
from jointformer.utils.data import write_dict_to_file


console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    filename=f"{os.environ.get('SLURM_JOB_NAME', 'run')}.log",
    filemode='a',
    format=f'{gethostname()}, rank {int(os.environ.get("SLURM_PROCID", 0))}: %(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Outputs directory
OUTPUTS_DIR: 'hyperparam_tuning_data/hyperparam_tuning_output'

# Path to hyperparameters grid in JSON format
HYPERPARAMETERS_GRID_FILEPATH: 'hyperparam_tuning_data/example_hyperparameters_grid.json'

# Here, setup fixed hyperparams for your training-evaluation procedure - single Optuna trial. Example hyperparams below
NUM_FOLDS: 5
NUM_EPOCHS: 10
BATCH_SIZE: 4
CROSS_VAL_VAL_SIZE: 0.2
NUM_WORKERS: 1
ACCELERATOR: "cpu"   # "gpu" or "tpu"

# Setup hyperparams corresponding to actual model's hyperparameters search
# Hyperparams for hyperparameter tuning
OPTUNA_METRIC_DIRECTION: 'minimize'   # 'minimize' or 'maximize'
OPTUNA_N_TRIALS: 2   # Number of trials to run
OPTUNA_N_JOBS: 1   # Number of parallel jobs
OPTUNA_SEED: 42

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='results')
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--fraction_train_dataset", type=float, default=1.)
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_seed", type=int, required=True)
    parser.add_argument("--model_seed", type=int, required=True)
    args = parser.parse_args()
    return args


def model_objective(trial, hyperparameters_grid, args):

    hyperparams = get_hparam_search_space(trial, hyperparameters_grid)

    # Example of how to access hyperparameters
    current_learning_rate = hyperparams['learning_rate']

    # Load configs
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_file(args.path_to_logger_config) if args.path_to_logger_config else None


    trainer_config.update(hyperparams) # print lr for sanit check return lr ** 2 - lr ** 4
    # Test
    if args.test:
        console.info("Running in test mode")
        trainer_config.max_iters = 2
        trainer_config.batch_size = 2
        trainer_config.eval_iters = 1
        trainer_config.eval_interval = 1
        trainer_config.log_interval = 1

    # Dump configs
    dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    # Init
    train_dataset = AutoDataset.from_config(dataset_config, split='train', data_dir=args.data_dir)
    num_subsamples =  int(len(train_dataset) * args.fraction_train_dataset)
    train_dataset._subset(num_samples=num_subsamples, seed=args.dataset_seed)
    console.info(f"Selected {len(train_dataset)} training examples")
    val_dataset = AutoDataset.from_config(dataset_config, split='val', data_dir=args.data_dir)
    trainer_config.correct_for_num_train_examples(num_train_examples=len(train_dataset))  # adjust trainer config to dataset size
    tokenizer = AutoTokenizer.from_config(tokenizer_config)

    model = AutoModel.from_config(model_config)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    trainer = Trainer(
        out_dir=args.out_dir, seed=args.model_seed, config=trainer_config, model=model,
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=val_dataset,
        tokenizer=tokenizer, logger=logger)

    if os.path.exists(args.path_to_model_ckpt):
        trainer.resume_from_file(args.path_to_model_ckpt)
        console.info(f"Resuming from {args.path_to_model_ckpt}")

    trainer.train()
    
    objective_metric = trainer.test()
    print(f"Objective metric: {objective_metric}")
    
    return objective_metric


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)

    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    # Create actual objective function using partial - pass in the hyperparameters grid
    objective_func = partial(model_objective, hyperparameters_grid=hyperparameters_grid, args=args)

    # Create a study object
    study = optuna.create_study(direction=OPTUNA_METRIC_DIRECTION, 
                                sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))   # Default sampler is TPESampler, providing seed for reproducibility

    # Start the hyperparameter tuning
    study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, n_jobs=OPTUNA_N_JOBS)
    study_df = study.trials_dataframe()

    # Save study dataframe
    study_df.to_csv(os.path.join(OUTPUTS_DIR, "study_results.csv"), index=False)
    # Save best params
    save_json(os.path.join(OUTPUTS_DIR, "best_params.json"), study.best_params)
           