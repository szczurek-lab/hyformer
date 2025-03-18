import os
import argparse 

from hyformer.utils.optuna import save_json
from experiments.fibrosis_prediction.train import main as model_training_loop


def main(args):

    if os.path.exists(os.path.join(args.out_dir, 'ckpt.pt')):    
        print(f"Model already trained.")
        
    else:
        # Create root directory
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Train Model
        _loss = model_training_loop(args)

        # Save the validation loss
        print(f"Validation loss with hparams: {_loss}")
        save_json(os.path.join(args.out_dir, "validation_loss_aggregated.json"), {"validation_loss": _loss})


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
    parser.add_argument("--model_seed", type=int, default=1337, help="Seed for the model")
    args = parser.parse_args()
    main(args)
