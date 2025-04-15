"""
Datasets for loading and processing data from various file formats.

This package provides classes for working with different types of datasets,
with a simple factory pattern for dataset instantiation from configuration files.

Classes
-------
AutoDataset
    Factory class for creating dataset instances based on configuration
BaseDataset
    Base class for all datasets, extending PyTorch's Dataset class
SequenceDataset
    Dataset for loading and processing sequence data from various file formats

Examples
--------
>>> from hyformer.utils.datasets import AutoDataset
>>> from hyformer.configs.dataset import DatasetConfig
>>> 
>>> # Create a dataset from configuration
>>> config = DatasetConfig(
...     dataset_type='sequence',
...     train_data_path='data/train.npz',
...     # Other config parameters...
... )
>>> dataset = AutoDataset.from_config(config, split='train')
"""
