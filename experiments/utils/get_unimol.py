import numpy as np

from hyformer.utils.file_io import load_lmdb_file, infer_string_dtype, load_npy_with_progress


def main():

    for split in ["train"]:
    
    
        output_filepath = f"/lustre/groups/aih/hyformer/data/unimol/{split}.npz"

        # Load dataset
        print("Loading data from", f"/lustre/groups/aih/hyformer/data/unimol/raw/{split}.lmdb")
        data = load_lmdb_file(
            lmdb_path=f"/lustre/groups/aih/hyformer/data/unimol/raw/{split}.lmdb",
            keys_to_extract=['smi'],
            max_readers=256,
            num_workers=32,
        )
        
        _data_dtype = infer_string_dtype(data)
        print(f"Data dtype: {_data_dtype}")
        print(f"Data shape: {len(data)}")
        
        
        target = load_npy_with_progress(
            f"/lustre/groups/aih/hyformer/data/unimol/raw/{split}_physchem.npy",
        )

        data = np.array(data, dtype=_data_dtype)
        target = np.array(target, dtype=np.float32).reshape(len(target), -1)
        
        assert len(data) == len(target), "Data and target must have the same length"
        if len(target.shape) == 1:
            target = target.reshape(-1, 1)
        assert len(target.shape) == 2, "Target must have 2 dimensions"
        assert np.isnan(target).sum() == 0, "Target must not contain NaNs"
        
        # Save to file
        np.savez(
            output_filepath,
            smiles=data,
            target=target,
            np_version=np.__version__
        )
        print(f"Saved {split} dataset to {output_filepath}")

if __name__ == "__main__":
    main()
