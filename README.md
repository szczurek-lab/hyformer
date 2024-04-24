# Jointformer

This is the official repository for the Jointformer model. 

Jointformer is a transformer-based generative model that can be jointly trained on generation and 
prediction tasks. 

## Getting Started

### Installation
To create an environment that satisfies the necessary requirements run
```
 conda env create -f env.yml
```
Next, install Jointformer from the project directory with 
```
conda activate hybrid-transformer
pip install -e .
```

Optionally, for a faster build use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or
enable [conda-libmamba-solver](https://www.anaconda.com/blog/conda-is-fast-now) with 
``` 
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Data

Data can be downloaded from ...

### Pre-trained Models

Pre-trained models can be downloaded from ...


## Experiments

### Model Training

To train a model run 
```
python experiments/joint_learning/train.py 
  --out_dir ./results/joint_learning/{MODEL}
  --path_to_model_config ./configs/models/{MODEL}
  --path_to_trained_config ./configs/trainers/joint_learning
```

### Pre-Training 

To pre-train a model run 
```
python scripts/pretrain/train.py 
  --out_dir ./results/pretrain/{MODEL}
  --path_to_model_config ./configs/models/{MODEL}
```
to eval
```
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/pretrain/eval.py --out_dir ./results/pretrain/{MODEL} --path_to_model_config ./configs/models/{MODEL}
```

### Joint Learning

Joint learning takes a pre-trained model and trains it jointly on generation and prediction tasks 

```
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/joint_learning/train.py 
  --out_dir ./results/joint_learning/
```

Evaluate with
```
CUDA_VISIBLE_DEVICES={GPU_ID} python scripts/joint_learning/evaluate.py 
  --out_dir ./results/joint_learning
  --data_reference_file ./data/guacamol/test/smiles.txt
```

```
CUDA_VISIBLE_DEVICES=3 python scripts/joint_learning/eval.py --out_dir /raid/aizd/hybrid_transformer/results/table_1/ --data_reference_file ./data/guacamol/test/smiles.txt
```

## Downloads

Download ZINC15 data from [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq)

## TODOs

  - scripts refer to names, which refer to benchmarks by a separate file linking configs to names
  - no other parameters should be passed to scripts or set in scripts manually
  - log console outputs and dump to file all the time 

## Results

## References
