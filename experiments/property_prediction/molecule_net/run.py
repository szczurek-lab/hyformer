""" Run property prediction experiments for MoleculeNet. """

import os
import argparse

import numpy as np

from hyformer.utils.file_io import save_json

from experiments.property_prediction.train import main as train
from experiments.property_prediction.test import main as test

RESULTS_FILENAME = "test_loss_aggregated.json"
SEED_ARRAY = [0, 1, 2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run property prediction experiments for MoleculeNet")
    parser.add_argument("--out_dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--data_dir", type=str, nargs='?', help="Path to the data directory")
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--tokenizer_config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--trainer_config_path", type=str, nargs='?')
    parser.add_argument("--generative_trainer_config_path", type=str, nargs='?')
    parser.add_argument("--predictive_trainer_config_path", type=str, nargs='?')
    parser.add_argument("--model_ckpt_path", type=str, nargs='?')
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--learning_rate", type=float, nargs='?')
    parser.add_argument("--batch_size", type=int, nargs='?')
    parser.add_argument("--max_epochs", type=int, nargs='?')
    parser.add_argument("--weight_decay", type=float, nargs='?')
    parser.add_argument("--decay_lr", type=lambda x: (str(x).lower() == 'true'), default=None, help="Dropout for the model")
    parser.add_argument("--dropout", type=float, nargs='?')
    parser.add_argument("--patience", type=int, nargs='?')
    args = parser.parse_args()
    return args
    
def main(args):
    
    # Create root directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)
    print(f"Saving results to: {args.out_dir}")

    # Check if experiment has already been run
    if os.path.exists(os.path.join(args.out_dir, RESULTS_FILENAME)):
        print("Experiment already run. Exiting...")
        return None

    # Train and test
    val_loss = np.zeros(shape=(len(SEED_ARRAY),))
    test_loss = np.zeros(shape=(len(SEED_ARRAY),))
    for seed in SEED_ARRAY:
        _out_dir = os.path.join(args.out_dir, f"seed_{seed}")
        
        if args.generative_trainer_config_path is not None:
            raise NotImplementedError("Generative training not implemented yet.")

        _val_loss = train(
            out_dir=_out_dir,
            data_dir=args.data_dir,
            experiment_seed=seed,
            dataset_config_path=args.dataset_config_path,
            tokenizer_config_path=args.tokenizer_config_path,
            model_config_path=args.model_config_path,
            trainer_config_path=args.predictive_trainer_config_path if args.predictive_trainer_config_path is not None else args.trainer_config_path,
            model_ckpt_path=args.model_ckpt_path,
            debug=args.debug,
            patience=args.patience,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            decay_lr=args.decay_lr,
            dropout=args.dropout
            )
        val_loss[seed] = _val_loss
        print(f"Val loss for seed {seed}: {_val_loss}")
        
        _test_loss = test(
            out_dir=_out_dir,
            data_dir=args.data_dir,
            experiment_seed=seed,
            dataset_config_path=args.dataset_config_path,
            tokenizer_config_path=args.tokenizer_config_path,
            model_config_path=args.model_config_path,
            trainer_config_path=args.predictive_trainer_config_path if args.predictive_trainer_config_path is not None else args.trainer_config_path,
            model_ckpt_path=args.model_ckpt_path
            )
        test_loss[seed] = _test_loss
        print(f"Test loss for seed {seed}: {_test_loss}")

    print(f"Test loss array: {test_loss}")
    print(f"Mean test loss: {np.mean(test_loss)}")
    print(f"Std test loss: {np.std(test_loss)}")
    print(f"Latex entry regression: {round(np.mean(test_loss), 3)}({round(np.std(test_loss), 3)})")
    print(f"Latex entry classification: {round(np.mean(test_loss) * 100, 1)}({round(np.std(test_loss) * 100, 1)})")

    # Save results
    save_json(os.path.join(args.out_dir, RESULTS_FILENAME), {"mean": np.mean(test_loss), "std": np.std(test_loss)})

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
