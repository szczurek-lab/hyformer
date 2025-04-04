#!/usr/bin/python3

from setuptools import setup, find_packages
from hyformer import __version__

setup(
    name="hyformer",
    version=__version__,
    packages=find_packages(),
    # Dependencies are managed through environment.yml
    # This allows for better conda/pip integration and CUDA support
    install_requires=[],
    author="Adam Izdebski",
    author_email="adam.izdebski1@gmail.com",
    description="An official implementation of Hyformer.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # Python version is managed through environment.yml (currently 3.10)
    # Keeping this for compatibility with pip installs outside of conda
    python_requires=">=3.10",
)
