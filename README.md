# Hybrid Transformer

## Setup

To create an environment that satisfies all the necessary requirements use
```
 conda env create -f env.yml
```
and install Hybrid Transformer within the environment, from the project directory, with 
```
conda activate hybrid-transformer
pip install -e .
```

Optionally, for a faster build, use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or
enable [conda-libmamba-solver](https://www.anaconda.com/blog/conda-is-fast-now) with 
``` 
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Tasks

### Distribution Learning

Distribution learning is an unsupervised pre-training task. Pre-train one of the three
available models `gpt`, `bert`, or `hybrid_transformer` with 
```
CUDA_VISIBLE_DEVICES={GPU_ID} python scripts/pretrain/train.py 
  --out_dir ./results/pretrain/{MODEL}
  --path_to_model_config ./configs/models/{MODEL}
```

### Joint Learning

Joint learning takes a pre-trained model and fine-tunes it on a prediction task. 

```
CUDA_VISIBLE_DEVICES={GPU_ID} python scripts/joint_learning/train.py 
  --out_dir ./results/joint_learning/
```

Evaluate the fine-tuned models with
```
CUDA_VISIBLE_DEVICES={GPU_ID} python scripts/joint_learning/evaluate.py 
  --out_dir ./results/joint_learning
  --data_reference_file ./data/guacamol/test/smiles.txt
```

```
CUDA_VISIBLE_DEVICES=3 python scripts/joint_learning/eval.py --out_dir /raid/aizd/hybrid_transformer/results/table_1/ --data_reference_file ./data/guacamol/test/smiles.txt
```



## Results

## References
