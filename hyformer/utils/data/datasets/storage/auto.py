import os
from typing import Dict, Any

from hyformer.utils.data.datasets.storage.base import DataStorage
from hyformer.utils.data.datasets.storage.npz import NPZStorage
from hyformer.utils.data.datasets.storage.csv import CSVStorage

# Default settings
_ALLOW_PICKLE = False  # Security setting - controls whether pickled objects are allowed in NPZ files


class AutoStorage:
    """Factory class for automatic storage backend selection based on file extension.

    This class provides factory methods for creating appropriate storage backends
    based on file extensions, allowing for convenient handling of different file formats.
    It automatically detects the file format and creates the appropriate storage backend.

    Currently supported formats:
    - .npz: NumPy compressed array files
    - .csv: Comma-separated values files
    """

    @classmethod
    def from_path(cls, filepath: str, **kwargs) -> DataStorage:
        """Create a storage backend for a file based on its extension.

        Parameters
        ----------
        filepath : str
            Path to the data file
        **kwargs
            Additional arguments to pass to the storage backend:
            - allow_pickle : bool, optional
              For NPZ files, controls whether to allow loading pickled objects
            - column_mapping : dict, optional
              For CSV files, maps logical column names to actual CSV column names
            - Additional pandas.read_csv parameters for CSV files

        Returns
        -------
        DataStorage
            Initialized storage backend

        Raises
        ------
        ValueError
            If the file format is not supported or if the file doesn't exist
        """
        # Validate filepath
        if not filepath or not os.path.isfile(filepath):
            raise ValueError(f"File not found: '{filepath}'")

        # Determine file type by extension
        if filepath.lower().endswith(".npz"):
            # Extract NPZ-specific options
            allow_pickle = kwargs.get("allow_pickle", _ALLOW_PICKLE)
            return NPZStorage.from_path(filepath, allow_pickle=allow_pickle)
        elif filepath.lower().endswith(".csv"):
            # Extract CSV-specific options
            csv_options = {
                k: v
                for k, v in kwargs.items()
                if k not in ["prediction_task_type", "num_prediction_tasks"]
            }
            return CSVStorage.from_path(filepath, **csv_options)
        else:
            raise ValueError(
                f"Unsupported file format: '{filepath}'. Supported formats: .npz, .csv"
            )

    @classmethod
    def get_supported_extensions(cls) -> Dict[str, str]:
        """Get a mapping of supported file extensions to their descriptions.

        Returns
        -------
        dict
            Dictionary mapping file extensions to descriptions
        """
        return {".npz": "NumPy compressed arrays", ".csv": "Comma-separated values"}
