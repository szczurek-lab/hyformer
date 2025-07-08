# Hyformer

The official implementation of [Hyformer](https://arxiv.org/abs/2310.02066), a [joint](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LasserreBishopMinka06.pdf) transformer-based model that simultaneously generates new molecules and predicts their properties.

Transformers are state-of-the-art models in both molecule generation and property prediction. However, generation and prediction are predominantly tackled in isolation, with separate models built for each of the tasks. Hyformer breaks this divide by unifying generation and prediction in a single architecture that combines a transformer encoder and decoder. This joint design leads to synergistic benefits: robust conditional generation, better out-of-distribution prediction, and meaningful representations.

<img src="_assets/hyformer.png" width="520" height="250"/>

## Installation

Install Hyformer directly from GitHub:
```
pip install git+https://github.com/adamizdebski/hyformer.git@jointformer-2.0#egg=hyformer
```

For local development with optional dependencies:
```
pip install -e .[<DEPENDENCIES>]
```

> Examples:
> ```
> pip install -e .                             # minimal installation, core dependencies
> pip install -e .[molecules]                  # molecule specific dependencies 
> pip install -e .[peptides]                   # peptide specific dependencies
> pip install -e .[dev]                        # dev specific dependencies
> pip install -e .[full]                       # all dependencies
> ```


### Installing with uv

To install Hyformer using [uv](https://docs.astral.sh/uv/), run
```
uv add git+https://github.com/adamizdebski/hyformer.git@jointformer-2.0#egg=hyformer
```

or for local development with optional dependencies:
```
uv pip install -e .[<DEPENDENCIES>]
```

### Installing with conda

To create an environment that satisfies the necessary requirements run
```
 conda env create -f env.yml
 conda activate hyformer
```

Optionally, for a faster build use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or
enable [conda-libmamba-solver](https://www.anaconda.com/blog/conda-is-fast-now) with 
``` 
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```
 
 > You can choose between `env_minimal.yml` for a minimal installation, `env.yml` for a minimal installation that includes chemical utils and `env_experiments.yml` for reproducing all experiments from the paper.


## Model Checkpoints

Pre-trained Hyformer models are available on Hugging Face:

### Molecular Models

| Model | Description | Hugging Face Link |
|-------|-------------|-------------------|
| **SMILES** | | |
| Hyformer-1M | A single layer model for debugging | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-base) |
| Hyformer-6M | Base model for fast generation and fine-tuning | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-property) |
| **SAFE** | | |
| Hyformer-1M-safe | Single layer SAFE model for debugging | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-selfies-1m) |

### Peptide Models

| Model | Description | Hugging Face Link |
|-------|-------------|-------------------|
| Hyformer-Peptide-Base | Base model trained on peptide sequences | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-peptide-base) |
| Hyformer-Peptide-Property | Joint peptide generation and property prediction | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-peptide-property) |
| Hyformer-Peptide-Large | Large-scale peptide model | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-peptide-large) |


> To log in to the Hugging Face Hub, run:
> ```
> huggingface-cli login
> huggingface-cli login --token $HUGGINGFACE_TOKEN
> ```

## Quickstart

To get started, load a pre-trained model from Hugging Face or a local directory:
```python
from hyformer import AutoTokenizer, AutoModel

model_name = "SzczurekLab/hyformer"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## Basic Usage

### Extract embeddings

### Unconditional generation

### Fine-tuning

### Conditional generation


----
## Reproducing Experiments

To reproduce the experiments from the paper, use
```
conda env create -f env_experiments.yml
conda activate hyformer_experiments
migrate_guacamol.sh  # enable distribution learning benchmarking with Gucacamol
```
In order to reproduce the experiments, an environment with additional dependencies is required.
To install the necessary dependencies, including [GuacaMol](https://github.com/BenevolentAI/guacamol)
 and [MoleculeNet](https://moleculenet.org/) benchmarks, run
```
conda env create --file hyformer-experiments.yml
```

For installing [MOSES](https://github.com/molecularsets/moses/tree/master), additionally run
```
git clone https://github.com/molecularsets/moses.git
cd moses
python setup.py install
```
and in case Git LFS is not enabled, manually substitute all data files in `moses/data/` and `moses/moses/dataset/data` directories.


### Representation Learning



## Contributiong to Hyformer


## License 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.0
as published by the Free Software Foundation.

## Citation

Cite our work with 
```
@misc{izdebski2025synergisticbenefits,
      title={Synergistic Benefits of Joint Molecule Generation and Property Prediction}, 
      author={Adam Izdebski and Jan Olszewski and Pankhil Gawade and Krzysztof Koras and Serra Korkmaz and Valentin Rauscher and Jakub M. Tomczak and Ewa Szczurek},
      year={2025},
      eprint={2504.16559},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.16559}, 
}
```


