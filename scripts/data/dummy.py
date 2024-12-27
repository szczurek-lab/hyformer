import os 
import argparse

import numpy as np

from jointformer.utils.data import save_strings_to_file, read_strings_from_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, nargs='+', required=True)
    parser.add_argument('--input_properties', type=str, required=False, default=None)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    smiles_all = []
    for input_data in args.input_data:
        # load data depending on file extension
        if input_data.endswith('.npy'):
            smiles = np.load(input_data).tolist()
        else:
            smiles = read_strings_from_file(input_data)
        smiles_all.extend(smiles)

    # create output dir, if it does not exist
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    save_strings_to_file(smiles_all, args.output)
    

if __name__ == '__main__':
    main()
    