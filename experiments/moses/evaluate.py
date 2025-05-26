# Imports
import argparse
import moses 
import pandas as pd
import numpy as np

from jointformer.utils.runtime import set_seed
from jointformer.utils.data import write_dict_to_file

def main(args):

    try:
        print("Loading train smiles")
        train = pd.read_csv(args.path_to_train_smiles)['SMILES'].tolist()
    except:
        print("No train smiles provided")
        train = None
    
    try:
        print("Loading test smiles")
        test = pd.read_csv(args.path_to_test_smiles)['SMILES'].tolist()
    except:
        print("No test smiles provided")
        test = None
    
    try:
        print("Loading test scaffolds")
        test_scaffolds = pd.read_csv(args.path_to_test_scaffolds)['SMILES'].tolist()
    except:
        print("No test scaffolds provided")
        test_scaffolds = None
    
    try:
        print("Loading test stats")
        test_stats = np.load(args.path_to_test_stats, allow_pickle=True)['stats'].item()
    except:
        print("No test stats provided")
        test_stats = None
    
    try:
        print("Loading test scaffold stats")
        test_scaffold_stats = np.load(args.path_to_test_scaffold_stats, allow_pickle=True)['stats'].item()
    except:
        print("No test scaffold stats provided")
        test_scaffold_stats = None

    generated_samples = pd.read_csv(args.path_to_generated_samples)['smiles'].tolist()
    metrics = moses.get_all_metrics(
        generated_samples,
        train=train,
        test=test,
        test_scaffolds=test_scaffolds,
        ptest=test_stats,
        ptest_scaffolds=test_scaffold_stats,
        batch_size=256,
        device="cuda:0",
    )
    print(metrics)
    write_dict_to_file(metrics, f"{args.out_dir}/moses_metrics.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Optuna Worker")
    parser.add_argument("--out_dir", type=str, required=True, help="Root directory for the experiment")
    parser.add_argument("--path_to_train_smiles", type=str, required=False, default=None)
    parser.add_argument("--path_to_test_smiles", type=str, required=False, default=None)
    parser.add_argument("--path_to_test_scaffolds", type=str, required=False, default=None)
    parser.add_argument("--path_to_test_stats", type=str, required=False, default=None)
    parser.add_argument("--path_to_test_scaffold_stats", type=str, required=False, default=None)
    parser.add_argument("--path_to_generated_samples", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=1337, help="Seed for random number generators")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
