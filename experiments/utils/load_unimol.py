#!/usr/bin/env python3
import argparse
import lmdb
import pickle
import numpy as np
import os
from tqdm import tqdm

def load_field_from_lmdb(lmdb_path, field):
    """
    Loads a single field from each record in the LMDB database and returns
    the collected values as a NumPy array.
    
    Assumes that each LMDB record is a pickled dictionary.
    """
    # Verify that the LMDB path is a directory.
    if not os.path.isdir(lmdb_path):
        raise ValueError(f"{lmdb_path} is not a directory. "
                         "LMDB data should be stored in a directory containing data.mdb and lock.mdb.")
    
    # Open the LMDB environment in read-only mode.
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True)
    data_list = []
    
    with env.begin(write=False) as txn:
        # Get the total number of entries (if available) for the progress bar.
        stat = txn.stat()
        total_entries = stat.get("entries", None)
        cursor = txn.cursor()
        
        for key, value in tqdm(cursor, total=total_entries, desc="Loading LMDB Records"):
            try:
                # Unpickle the stored value.
                record = pickle.loads(value)
            except Exception as e:
                print(f"Error unpickling record with key {key}: {e}")
                continue

            # Extract the desired field if it exists.
            if field in record:
                data_list.append(record[field])
            else:
                print(f"Warning: field '{field}' not found in record with key: {key}")
    
    return np.array(data_list)

def main():
    parser = argparse.ArgumentParser(
        description="Load a single field from a pickled LMDB and store the results in a NumPy array."
    )
    parser.add_argument("lmdb_path", help="Path to the LMDB directory (should contain data.mdb, lock.mdb, etc.)")
    parser.add_argument("field", help="Name of the field to extract from each record")
    args = parser.parse_args()

    try:
        data_array = load_field_from_lmdb(args.lmdb_path, args.field)
    except Exception as e:
        print("Error loading LMDB data:", e)
        return

    print("Loaded data array with shape:", data_array.shape)
    
    # Optionally, save the array for later use.
    output_file = f"{args.field}_output.npy"
    np.save(output_file, data_array)
    print(f"Data array saved to {output_file}")

if __name__ == "__main__":
    main()
