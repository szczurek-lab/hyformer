import os
import argparse 

import pandas as pd
from jointformer.utils.optuna import save_json
from experiments.fibrosis_prediction.train import main as model_training_loop


def main(args):

    _validation_loss_filename = os.path.join(args.data_dir, 'validation_loss_aggregated.json')
    _aggregated_loss_filename = os.path.join(args.out_dir, 'aggregated_loss.csv')

    # load the validation loss
    if os.path.exists(_validation_loss_filename):
        _df = pd.read_json(_validation_loss_filename)
        _loss = _df['validation_loss'].values[0]
        print(f"Validation loss with hparams: {_loss}")
        _values = [args.lr, args.batch_size, _loss]
        _df = pd.DataFrame([_values], columns=['lr', 'bs', 'validation_loss'])
    else:
        raise FileNotFoundError(f"Validation loss file not found at: {_validation_loss_filename}")
    
    # check if file _aggregated_loss_filename exists
    if os.path.exists(_aggregated_loss_filename):
        df = pd.read_csv(_aggregated_loss_filename)
    else:
        df = pd.DataFrame(columns=['lr', 'bs', 'validation_loss'])

    # append the new values
    df = df.append(_df, ignore_index=True)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)
        
    pd.save_csv(_aggregated_loss_filename, df)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hi experiment.")
    parser.add_argument("--out-dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for the model")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for the model")   
    args = parser.parse_args()
    main(args)
