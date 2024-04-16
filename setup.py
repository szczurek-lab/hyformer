#!/usr/bin/python3

from setuptools import setup, find_packages


setup(
    name='Jointformer',
    version='1.0.0',
    author='Adam Izdebski',
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=[],
    zip_safe=False
)
