import pandas as pd
import numpy as np
import os
from typing import Any, Optional, List, Union, Dict

from hyformer.utils.data.datasets.storage.base import DataStorage


class CSVStorage(DataStorage):
    """Storage backend for CSV files.

    This class provides functionality to load data from CSV files for use in datasets.
    It supports loading both input features and target values from the same CSV file,
    with optional column mapping for logical vs. actual column names.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Pandas DataFrame containing the loaded CSV data
    column_mapping : dict, optional
        Dictionary mapping logical column names to actual CSV column names

    Examples
    --------
    >>> import pandas as pd
    >>> from hyformer.utils.data.datasets.storage.csv_storage import CSVStorage
    >>>
    >>> # Create a storage instance from a DataFrame
    >>> df = pd.DataFrame({
    ...     'smiles': ['CC', 'CCC', 'CCCC'],
    ...     'activity': [0.5, 1.2, 0.8]
    ... })
    >>> storage = CSVStorage(df)
    >>>
    >>> # Load data and target
    >>> smiles = storage.load_data('smiles')
    >>> activity = storage.load_target('activity')
    """

    def __init__(
        self, dataframe: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None
    ):
        """Initialize CSV storage with a pandas DataFrame.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Pandas DataFrame containing the CSV data
        column_mapping : dict, optional
            Optional mapping from logical names to actual column names

        Raises
        ------
        ValueError
            If the dataframe is empty or not a pandas DataFrame
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("dataframe must be a pandas DataFrame")
        if dataframe.empty:
            raise ValueError("dataframe cannot be empty")

        self.dataframe = dataframe
        self.column_mapping = column_mapping or {}

    def _get_column_name(self, key: str) -> str:
        """Map a logical column name to an actual column name.

        Parameters
        ----------
        key : str
            Logical column name

        Returns
        -------
        str
            Actual column name in the DataFrame
        """
        return self.column_mapping.get(key, key)

    def load_data(self, data_key: str) -> np.ndarray:
        """Load input data from the CSV file.

        Parameters
        ----------
        data_key : str
            The column name containing input data (e.g., 'smiles')

        Returns
        -------
        numpy.ndarray
            NumPy array of input data values

        Raises
        ------
        ValueError
            If the column doesn't exist in the CSV or the key is invalid
        """
        if not data_key or not isinstance(data_key, str):
            raise ValueError(f"Invalid data_key: '{data_key}'")

        column = self._get_column_name(data_key)
        if column not in self.dataframe.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")
        return self.dataframe[column].values

    def load_target(self, target_key: Union[str, List[str]]) -> Optional[np.ndarray]:
        """Load target values from the CSV file.

        Parameters
        ----------
        target_key : str or list of str
            Column name(s) containing target values.
            Can be a single string or list of strings for multiple targets

        Returns
        -------
        numpy.ndarray or None
            NumPy array of target values, or None if target_key is not found

        Raises
        ------
        ValueError
            If any of the requested columns don't exist
        """
        if target_key is None:
            return None

        # Handle multiple target columns
        if isinstance(target_key, list):
            if not target_key:
                return None  # Empty list case

            columns = [self._get_column_name(k) for k in target_key]
            missing = [col for col in columns if col not in self.dataframe.columns]
            if missing:
                raise ValueError(f"Target columns {missing} not found in CSV file")
            return self.dataframe[columns].values

        # Handle single target column
        column = self._get_column_name(target_key)
        if column not in self.dataframe.columns:
            return None
        return self.dataframe[column].values.reshape(-1, 1)

    @classmethod
    def from_path(
        cls, path: str, column_mapping: Optional[Dict[str, str]] = None, **kwargs
    ) -> "CSVStorage":
        """Create a CSVStorage instance from a file path.

        Parameters
        ----------
        path : str
            Path to the CSV file
        column_mapping : dict, optional
            Optional mapping from logical names to CSV column names
        **kwargs
            Additional arguments to pass to pandas.read_csv

        Returns
        -------
        CSVStorage
            Initialized CSVStorage instance

        Raises
        ------
        ValueError
            If the file cannot be loaded or is invalid
        FileNotFoundError
            If the specified file cannot be found
        """
        try:
            # Validate path
            if not path or not isinstance(path, str):
                raise ValueError(f"Invalid path: '{path}'")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"CSV file not found: '{path}'")

            # Use pandas to load the CSV file
            dataframe = pd.read_csv(path, **kwargs)

            # Check if the DataFrame has data
            if dataframe.empty:
                raise ValueError(f"CSV file '{path}' is empty")

            return cls(dataframe, column_mapping)
        except FileNotFoundError as e:
            raise e
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file '{path}' is empty or has no columns")
        except Exception as e:
            raise ValueError(f"Failed to load CSV data from '{path}': {str(e)}")
