import optuna
import argparse
import random
import time 
import os

def objective(trial):
    seed = trial.suggest_int("seed", 1, 100) # sample a random seed
    random.seed(seed) # set the seed for reproducibility
    time.sleep(100) # simulate a long-running computation
    random_number = random.random() # generate a random number
    return random_number


def main(root_dir, study_name, direction, storage, optuna_n_trials, optuna_n_jobs, seed):
    
    # Set the root directory
    os.makedirs(root_dir, exist_ok=True)
    
    # Attempt to load the study; if it doesn't exist, create it
    storage = storage if storage is not None else f"sqlite:///optuna_studies/optuna_study_{study_name}.db"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        # Study does not exist, so we create it
        study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, sampler=optuna.samplers.TPESampler(seed=args.seed))

    # Run the optimization
    study.optimize(objective, n_trials=optuna_n_trials, n_jobs=optuna_n_jobs)

    # Save the study
    study.trials_dataframe().to_csv(os.path.join(root_dir, f"{study_name}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Optuna Worker")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--study-name", type=str, required=True, help="Name of the Optuna study")
    parser.add_argument("--direction", type=str, required=True, help="Direction of the optimization")
    parser.add_argument("--storage", type=str, default=None, help="Database URL for Optuna study")
    parser.add_argument("--optuna-n-trials", type=int, default=None, help="Number of trials to run in this instance")
    parser.add_argument("--optuna-n-jobs", type=int, default=1, help="Number of parallel jobs to run")
    parser.add_argument("--seed", type=int, default=0, help="Seed for Optuna study")
    args = parser.parse_args()

    main(
        study_name=args.study_name, direction=args.direction, storage=args.storage, optuna_n_trials=args.optuna_n_trials,
        optuna_n_jobs=args.optuna_n_jobs, seed=args.seed, root_dir=args.root_dir)
