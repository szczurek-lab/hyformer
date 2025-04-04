from hyformer.utils.properties.auto import AutoTarget
import argparse
import multiprocessing as mp
import numpy as np
import os
import math

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--input_properties_path", type=str, required=True)
    parser.add_argument("--output_data_path", type=str, required=True)
    parser.add_argument("--output_properties_path", type=str, required=True)
    return parser


def filter_nans(data_path, properties_path):

    data = np.load(data_path)
    properties = np.load(properties_path)
    print("Loaded dtypes: ", data.dtype, properties.dtype)
    assert data.shape[0] == properties.shape[0]
    
    mask = ~np.any(np.isnan(properties), axis=1)
    return data[mask], properties[mask]     


def main(args):
    dname = os.path.dirname(args.output_data_path)
    os.makedirs(dname, exist_ok=True)

    dname = os.path.dirname(args.output_properties_path)
    os.makedirs(dname, exist_ok=True)

    data, properties = filter_nans(args.input_data_path, args.input_properties_path)

    assert data.shape[0] == properties.shape[0]

    np.save(args.output_data_path, data)
    np.save(args.output_properties_path, properties)
    print(f"Extracted {data.shape[0]} samples.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    