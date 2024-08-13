import argparse
import os
import random 

from jointformer.utils.data import read_strings_from_file, save_strings_to_file


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(args):
    dname = os.path.dirname(args.output)
    os.makedirs(dname, exist_ok=True)

    random.seed(args.seed)

    data = read_strings_from_file(args.data_path)
    data = random.sample(data, args.num_samples)  # sample without replacement
    save_strings_to_file(data, args.output)
    
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
