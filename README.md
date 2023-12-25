# Hybrid Transformer

## Setup

To create and activate an environment that satisfies all the necessary requirements use
```
 conda env create -f env.yml
 conda activate hybrid-transformer
```
and install Hybrid Transformer, from the project directory, with 
```
pip install -e .
```

Optionally, for a faster build, use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or
enable [conda-libmamba-solver](https://www.anaconda.com/blog/conda-is-fast-now) with 
``` 
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Data

Data is available at ...

## Model

### Pre-Train

To train a model on a single GPU, run
```
python train.py
```
To run with multiple GPUs, run
```
torchrun train.py
```

### Fine-Tune

### Sample

## Results

## References
