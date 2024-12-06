""" Load and savews data. """

import os
import pandas as pd
import numpy as np

from typing import Union, List


def load(file_path: str, **kwargs) -> Union[np.ndarray, pd.DataFrame, List[str]]:

    """ Load data from a file.

    Args:
        file_path: Path to the file to load.

    Returns:
        pd.DataFrame: Data loaded from the file.

    """

    # Check if the file exists
    assert os.path.exists(file_path) and os.path.isfile(file_path), f"File {file_path} does not exist."
    
    # Load the data, depending on the file extension
    _extension = _get_file_extension(file_path)
    if _extension == '.csv':
        data = pd.read_csv(file_path, **kwargs)
    elif _extension == '.gz':
        data = pd.read_csv(file_path, compression='gzip', **kwargs)
    elif _extension == '.npy':
        data = np.load(file_path, **kwargs)
    elif _extension == '.txt' or _extension == '.smiles':
        data = np.loadtxt(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {_extension}")
    
    return data


def save(data: Union[np.ndarray, pd.DataFrame, List[str]], file_path: str, overrite: bool = False) -> None:

    """ Save data to a file.

    Args:
        data: Data to save.
        file_path: Path to the file to save the data to.

    """

    # Check if the file exists and create the directory if it does not
    if not overrite:
        assert not os.path.exists(file_path), f"File {file_path} already exists."

    dname = os.path.dirname(file_path)
    if not os.path.exists(dname):
        os.makedirs(dname, exist_ok=False)

    # Save the data, depending on the file extension
    _extension = _get_file_extension(file_path)
    if _extension == '.csv':
        data.to_csv(file_path, index=False)
    elif _extension == '.gz':
        data.to_csv(file_path, compression='gzip', index=False)
    elif _extension == '.npy':
        np.save(file_path, data)
    elif _extension == '.txt':
        np.savetxt(file_path, data)
    else:
        raise ValueError(f"Unsupported file extension: {_extension}")
    
    return None


def _get_file_extension(file_path: str) -> str:

    """ Get the file extension.

    Args:
        file_path: Path to the file.

    Returns:
        str: File extension.

    """
    return os.path.splitext(file_path)[1]
