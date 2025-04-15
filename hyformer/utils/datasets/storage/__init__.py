"""
Storage backends for loading data from different file formats.

This package provides interfaces and implementations for loading data
from various file formats like NPZ, CSV, etc. using a simple factory pattern
based on file extensions.

Classes
-------
DataStorage
    Abstract base class defining the interface for all storage backends
AutoStorage
    Factory class for creating appropriate storage backends based on file extension
NPZStorage
    Storage backend for NumPy's .npz compressed array files
CSVStorage
    Storage backend for CSV files using pandas

Examples
--------
>>> from hyformer.utils.datasets.storage.auto import AutoStorage
>>> 
>>> # Load data from a file based on its extension
>>> storage = AutoStorage.from_path('data/molecules.npz')
>>> 
>>> # Load input and target data
>>> smiles = storage.load_data('smiles')
>>> properties = storage.load_target('properties')
"""
