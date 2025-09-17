import numpy as np
from typing import Any, Optional, List, Union, Dict, TypeVar

from hyformer.utils.data.datasets.storage.base import DataStorage

# Type alias for NumPy npz file type
NpzFile = TypeVar("NpzFile", bound="np.lib.npyio.NpzFile")


class NPZStorage(DataStorage):
    """Storage backend for NPZ files.

    This class provides functionality to load data from NumPy's .npz compressed
    archive format for use in datasets. It supports loading both input data and
    target values from the same NPZ file.

    Parameters
    ----------
    data_file : np.lib.npyio.NpzFile
        NumPy NPZ file object containing arrays
    allow_pickle : bool, default=False
        Whether to allow loading pickled objects (security consideration)
    """

    def __init__(self, data_file: NpzFile, allow_pickle: bool = False):
        """Initialize NPZ storage with a NumPy NPZ file.

        Parameters
        ----------
        data_file : np.lib.npyio.NpzFile
            NumPy NPZ file object containing arrays
        allow_pickle : bool, default=False
            Whether to allow loading pickled objects
        """
        self.data_file = data_file
        self.allow_pickle = allow_pickle

    def load_data(self, data_key: str) -> np.ndarray:
        """Load input data from the NPZ file.

        Parameters
        ----------
        data_key : str
            The array name containing input data

        Returns
        -------
        numpy.ndarray
            NumPy array of input data

        Raises
        ------
        ValueError
            If the key doesn't exist in the NPZ file
        """
        if data_key not in self.data_file:
            raise ValueError(f"Key '{data_key}' not found in NPZ file")
        return self.data_file[data_key]

    def load_target(self, target_key: Union[str, List[str]]) -> Optional[np.ndarray]:
        """Load target values from the NPZ file.

        Parameters
        ----------
        target_key : str or list of str
            Array name(s) containing target values.
            Can be a single string or list of strings for multiple targets

        Returns
        -------
        numpy.ndarray or None
            NumPy array of target values, or None if target_key is not found

        Raises
        ------
        ValueError
            If any of the requested keys don't exist
        """
        if target_key is None:
            return None

        # Handle multiple target keys
        if isinstance(target_key, list):
            missing = [k for k in target_key if k not in self.data_file]
            if missing:
                raise ValueError(f"Target keys {missing} not found in NPZ file")
            # Create a structured array combining all target arrays
            return np.column_stack([self.data_file[k] for k in target_key])

        # Handle single target key
        if target_key not in self.data_file:
            return None
        return self.data_file[target_key]

    @classmethod
    def from_path(cls, path: str, allow_pickle: bool = False, **kwargs) -> "NPZStorage":
        """Create an NPZStorage instance from a file path.

        Parameters
        ----------
        path : str
            Path to the NPZ file
        allow_pickle : bool, default=False
            Whether to allow loading pickled objects
        **kwargs
            Additional arguments (not used for NPZ files)

        Returns
        -------
        NPZStorage
            Initialized NPZStorage instance

        Raises
        ------
        ValueError
            If the file cannot be loaded or doesn't exist
        FileNotFoundError
            If the specified file cannot be found
        """
        try:
            # Validate path
            if not path or not isinstance(path, str):
                raise ValueError(f"Invalid path: '{path}'")

            data_file = np.load(path, allow_pickle=allow_pickle)
            return cls(data_file, allow_pickle)
        except FileNotFoundError:
            raise FileNotFoundError(f"NPZ file not found: '{path}'")
        except Exception as e:
            raise ValueError(f"Failed to load NPZ data from '{path}': {str(e)}")
